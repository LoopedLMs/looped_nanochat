"""
GPT model (rewrite, a lot simpler)
Notable features:
- rotary embeddings (and no positional embeddings)
- QK norm
- untied weights for token embedding and lm_head
- relu^2 activation in MLP
- sandwich norm for stable recurrence
- RMSNorm with learnable scale
- no bias in linear layers
- Group-Query Attention (GQA) support for more efficient inference
- Flash Attention 3 integration

Looped Transformer specifics:
- Three-stage architecture: prelude (n_prelude layers) → recur (n_recur_block layers, run num_recur times) → coda (n_coda layers)
- Input injection modes (input_injection config):
  - "inject_init_prelude": u = inject(concat(e, s)) with s initially from prelude output (default looped behavior)
  - "inject_init_random": u = inject(concat(e, s)) with s initially sampled from N(0, 1/sqrt(d))
  - "passthrough": u = s (pure recurrence, no inject layer), s initially from prelude output
  - inject layer initialized as identity-like [I|0] so inject(concat(e,s)) ≈ e initially
- Warm-start inference: final recurrent state s from token t-1 initializes state for token t (disabled for training)
- Truncated BPTT: gradients detached for all but last bptt_k recurrences to limit memory/compute
- Variable recurrence depth: num_recur sampled from distribution during training (mean=train_recur_mean, max=train_recur_max)
"""

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.common import get_dist_info, print0

# Our custom Flash Attention module that automatically uses FA3 on Hopper+ and SDPA fallback elsewhere
from nanochat.flash_attention import flash_attn
from nanochat.gates import GateChecker, GateConfig, GateStats
from nanochat.optim import DistMuonAdamW, MuonAdamW


@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 65536
    size: int = 20  # Model size knob: model_dim = size * aspect_ratio
    n_head: int = 10  # number of query heads
    n_kv_head: int = 10  # number of key/value heads (GQA)
    n_embd: int = 1280
    # Sliding window attention pattern string, tiled across layers. Final layer always L.
    # Characters: L=long (full context), S=short (half context)
    # Examples: "L"=all full context, "SL"=alternating, "SSL"=two short then one long
    window_pattern: str = "LLSSSLLL"

    # Looped Transformer config options
    n_prelude: int = 2
    n_recur_block: int = 4
    n_coda: int = 2

    # Looped Transformer training options
    train_recur_mean: float = 4.0  # mean recurrences during training (samples from distribution)
    train_recur_max: int = 16  # max recurrences sampled during training
    bptt_k: int = 4  # truncate backprop to last k recurrences
    # Input injection mode: controls how recurrent state is initialized and injected
    # - "inject_init_prelude": inject(concat(e, s)) with s initially from prelude output (default looped behavior)
    # - "inject_init_random": inject(concat(e, s)) with s initially sampled from N(0, 1/sqrt(d))
    # - "passthrough": no injection, s passes through directly (pure recurrence, s initially from prelude)
    input_injection: Literal["inject_init_prelude", "inject_init_random", "passthrough"] = "inject_init_prelude"
    logit_softcap: float = 15.0  # smoothly cap logits to [-softcap, softcap] via tanh

    def __post_init__(self):
        valid_modes = {"inject_init_prelude", "inject_init_random", "passthrough"}
        if self.input_injection not in valid_modes:
            raise ValueError(f"input_injection must be one of {valid_modes}, got {self.input_injection}")


def norm(x, eps: float = 1e-6):
    # Purely functional rmsnorm with no learnable params (used for QK norm)
    return F.rms_norm(x, (x.size(-1),), eps=eps)


class RMSNorm(nn.Module):
    """RMSNorm with learnable scale parameter."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.rms_norm(x, (x.size(-1),), eps=self.eps) * self.weight


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4  # multihead attention
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]  # split up last dim into two halves
    y1 = x1 * cos + x2 * sin  # rotate pairs of dims
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)

    def forward(self, x, cos_sin, window_size, kv_cache, layer_idx=None):
        B, T, C = x.size()
        # Fail fast: layer_idx required when using kv_cache
        assert (kv_cache is None) or (layer_idx is not None), "layer_idx required when kv_cache is provided"

        # Project the input to get queries, keys, and values
        # Shape: (B, T, H, D) - FA3's native layout, no transpose needed!
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # Apply Rotary Embeddings to queries and keys to get relative positional encoding
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)  # QK norm

        # Flash Attention (FA3 on Hopper+, PyTorch SDPA fallback elsewhere)
        # window_size is (left, right) tuple: (N, 0) for causal, (-1, 0) for full context
        if kv_cache is None:
            # Training: causal attention with optional sliding window
            y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        else:
            # Inference: use flash_attn_with_kvcache which handles cache management
            k_cache, v_cache = kv_cache.get_layer_cache(layer_idx)
            y = flash_attn.flash_attn_with_kvcache(
                q,
                k_cache,
                v_cache,
                k=k,
                v=v,
                cache_seqlens=kv_cache.cache_seqlens,
                causal=True,
                window_size=window_size,
            )

        # Re-assemble the heads and project back to residual stream
        y = y.contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)
        # Sandwich norm layers: n1/n2 for attention, n3/n4 for MLP
        self.n1 = RMSNorm(config.n_embd)
        self.n2 = RMSNorm(config.n_embd)
        self.n3 = RMSNorm(config.n_embd)
        self.n4 = RMSNorm(config.n_embd)

    def forward(self, x, cos_sin, window_size, kv_cache, layer_idx=None):
        # Sandwich norm format: x̂=n2(x+Attn(n1(x))), x=n4(x̂+MLP(n3(x̂)))
        x = self.n2(x + self.attn(self.n1(x), cos_sin, window_size, kv_cache, layer_idx))
        x = self.n4(x + self.mlp(self.n3(x)))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig, pad_vocab_size_to: int = 64):
        """
        NOTE a major footgun: this __init__ function runs in meta device context (!!)
        Therefore, any calculations inside here are shapes and dtypes only, no actual data.
        => We actually initialize all data (parameters, buffers, etc.) in init_weights() instead.
        """
        super().__init__()
        self.config = config

        # Compute per-layer window sizes for sliding window attention
        # window_size is (left, right) tuple: (-1, 0) for full context, (N, 0) for sliding window
        self.window_sizes = self._compute_window_sizes(config)
        # Pad vocab for efficiency (DDP, tensor cores). This is just an optimization - outputs are cropped in forward().
        # https://huggingface.co/docs/transformers/main_classes/model#transformers.PreTrainedModel.resize_token_embeddings
        padded_vocab_size = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        if padded_vocab_size != config.vocab_size:
            print0(f"Padding vocab_size from {config.vocab_size} to {padded_vocab_size} for efficiency")
        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(padded_vocab_size, config.n_embd),
                "prelude": nn.ModuleList([Block(config) for _ in range(config.n_prelude)]),
                "recur": nn.ModuleList([Block(config) for _ in range(config.n_recur_block)]),
                "coda": nn.ModuleList([Block(config) for _ in range(config.n_coda)]),
            }
        )
        # Input injection adapter (only needed when not using passthrough mode)
        if config.input_injection != "passthrough":
            self.inject = nn.Linear(2 * config.n_embd, config.n_embd, bias=False)
        # RMSNorm layers outside blocks
        self.norm_emb = RMSNorm(config.n_embd)  # after embedding
        self.norm_recur = RMSNorm(config.n_embd)  # at end of recurrent block (nc)
        self.norm_final = RMSNorm(config.n_embd)  # before lm_head
        self.lm_head = nn.Linear(config.n_embd, padded_vocab_size, bias=False)
        # To support meta device initialization, we init the rotary embeddings here, but it's just "fake" meta tensors only.
        # As for rotary_seq_len, these rotary embeddings are pretty small/cheap in memory,
        # so let's just over-compute them by 10X, but assert fail if we ever reach that amount.
        # In the future we can dynamically grow the cache, for now it's fine.
        self.rotary_seq_len = config.sequence_len * 10  # 10X over-compute should be enough, TODO make nicer?
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)  # persistent=False means it's not saved to the checkpoint
        self.register_buffer("sin", sin, persistent=False)

    @torch.no_grad()
    def init_weights(self):
        """
        Initialize the full model in this one function for maximum clarity.

        wte (embedding):     normal, std=1.0
        lm_head:             normal, std=0.001
        for each block:
            attn.c_q:        uniform, std=1/sqrt(n_embd)
            attn.c_k:        uniform, std=1/sqrt(n_embd)
            attn.c_v:        uniform, std=1/sqrt(n_embd)
            attn.c_proj:     zeros
            mlp.c_fc:        uniform, std=1/sqrt(n_embd)
            mlp.c_proj:      zeros
        """

        # Embedding and unembedding
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)

        # Transformer blocks: uniform init with bound = sqrt(3) * std (same standard deviation as normal)
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5  # sqrt(3) multiplier makes sure Uniform achieves the same std as Normal
        for block in list(self.transformer.prelude) + list(self.transformer.recur) + list(self.transformer.coda):
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)  # weights use Uniform to avoid outliers
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight)  # projections are zero
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)

        # Initialize inject layer as identity-like: output = e (first half of concat(e, s))
        # This ensures gradients flow on the first forward pass
        # Weight shape is (n_embd, 2*n_embd), we want [I | 0] so inject(concat(e,s)) ≈ e
        # Only initialize if inject layer exists (not in passthrough mode)
        if self.config.input_injection != "passthrough":
            with torch.no_grad():
                self.inject.weight.zero_()
                self.inject.weight[:, :n_embd].copy_(torch.eye(n_embd))

        # RMSNorm weights: initialize to ones
        for module in self.modules():
            if isinstance(module, RMSNorm):
                torch.nn.init.ones_(module.weight)

        # Rotary embeddings
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin

        # Cast embeddings to bf16: optimizer can tolerate it and it saves memory
        if self.transformer.wte.weight.device.type == "cuda":
            self.transformer.wte.to(dtype=torch.bfloat16)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
        # TODO: bump base theta more? e.g. 100K is more common more recently
        # autodetect the device from model embeddings
        if device is None:
            device = self.transformer.wte.weight.device
        # stride the channels
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        # stride the time steps
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        # calculate the rotation frequencies at each (time, channel) pair
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.bfloat16(), sin.bfloat16()  # keep them in bfloat16
        cos, sin = (
            cos[None, :, None, :],
            sin[None, :, None, :],
        )  # add batch and head dims for later broadcasting
        return cos, sin

    def _compute_window_sizes(self, config: GPTConfig):
        """
        Compute per-layer window sizes for sliding window attention.

        Returns list of (left, right) tuples for FA3's window_size parameter:
        - left: how many tokens before current position to attend to (-1 = unlimited)
        - right: how many tokens after current position to attend to (0 for causal)

        Pattern string is tiled across layers. Final layer always gets L (full context).
        Characters: L=long (full context), S=short (half context)
        """
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern), f"Invalid window_pattern: {pattern}. Use only S and L."
        # Map characters to window sizes
        long_window = config.sequence_len
        short_window = long_window // 2
        char_to_window = {
            "L": (long_window, 0),
            "S": (short_window, 0),
        }
        # Tile pattern across layers
        window_sizes = []
        for layer_idx in range(config.n_prelude + config.n_recur_block + config.n_coda):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        # Final layer always gets full context
        window_sizes[-1] = (long_window, 0)
        return window_sizes

    def get_device(self):
        return self.transformer.wte.weight.device

    def _get_kv_layer_idx(
        self,
        section: str,
        block_idx: int,
        kv_budget: int | None,
        recur_iter: int | None = None,
    ) -> int | None:
        """
        Compute the KV cache layer index for a given section and block.

        The KV cache layout is:
            [prelude layers | recurrence slots | coda layers]
            [0..n_prelude)  | [n_prelude..n_prelude + kv_budget*n_recur_block) | [coda start..)

        Args:
            section: One of "prelude", "recur", or "coda"
            block_idx: Index within the section (0-indexed)
            kv_budget: Number of recurrence slots to store in cache (None means no cache)
            recur_iter: Current recurrence iteration (required for "recur" section)

        Returns:
            The layer index for the KV cache, or None if kv_budget is None
        """
        if kv_budget is None:
            return None

        if section == "prelude":
            return block_idx
        elif section == "recur":
            assert recur_iter is not None, "recur_iter required for recur section"
            # Circular indexing: slot = recur_iter % kv_budget
            slot = recur_iter % kv_budget
            return self.config.n_prelude + (slot * self.config.n_recur_block) + block_idx
        elif section == "coda":
            # Coda comes after all recurrence slots
            return self.config.n_prelude + (kv_budget * self.config.n_recur_block) + block_idx
        else:
            raise ValueError(f"Unknown section: {section}")

    def estimate_flops(self, num_recur=None):
        """
        Return the estimated FLOPs per token for the looped model (forward + backward).

        Each matmul weight parameter contributes 2 FLOPs (multiply *, accumulate +) in forward, and 2X that in backward => 2+4=6.
        Cleanest explanation of this: https://medium.com/@dzmitrybahdanau/the-flops-calculus-of-language-model-training-3b19c1f025e4
        On top of that, 12 * h * q * effective_seq_len accounts for key @ query matmul flops inside attention.
        With sliding windows, effective_seq_len varies per layer (capped by window size).
        Ref: https://arxiv.org/abs/2204.02311 (PaLM paper).
        This is ~1% off from the exact formulas of Chinchilla paper, the difference is:
        - Chinchilla counts the embedding layer as flops (? weird, it's just a lookup => we ignore)
        - Chinchilla counts exp/sum/divide in attention softmax as flops (a little sus and very tiny => we ignore)

        Args:
            num_recur: Number of recurrences to assume. If None, uses train_recur_mean.
        """
        if num_recur is None:
            num_recur = int(self.config.train_recur_mean)

        h, q, t = (
            self.config.n_head,
            self.config.n_embd // self.config.n_head,
            self.config.sequence_len,
        )

        # 1. Count parameters by section
        prelude_params = sum(p.numel() for p in self.transformer.prelude.parameters())
        recur_params = sum(p.numel() for p in self.transformer.recur.parameters())
        coda_params = sum(p.numel() for p in self.transformer.coda.parameters())
        inject_params = sum(p.numel() for p in self.inject.parameters()) if self.config.input_injection != "passthrough" else 0
        lm_head_params = sum(p.numel() for p in self.lm_head.parameters())

        # 2. Matmul FLOPs weighted by usage
        matmul_flops = 6 * (
            prelude_params  # prelude runs 1x
            + recur_params * num_recur  # recur runs num_recur times
            + coda_params  # coda runs 1x
            + inject_params * num_recur  # inject runs num_recur times (0 if passthrough)
            + lm_head_params  # lm_head runs 1x
        )

        # 3. Attention FLOPs weighted by usage
        attn_flops = 0
        num_layers = self.config.n_prelude + self.config.n_recur_block + self.config.n_coda
        for layer_idx in range(num_layers):
            window = self.window_sizes[layer_idx][0]
            effective_seq = min(window, t)

            # Determine how many times this layer runs
            if layer_idx < self.config.n_prelude:
                multiplier = 1  # prelude
            elif layer_idx < self.config.n_prelude + self.config.n_recur_block:
                multiplier = num_recur  # recur
            else:
                multiplier = 1  # coda

            attn_flops += 12 * h * q * effective_seq * multiplier

        num_flops_per_token = matmul_flops + attn_flops
        return num_flops_per_token

    def num_scaling_params(self):
        """
        Return detailed parameter counts for scaling law analysis.
        Different papers use different conventions:
        - Kaplan et al. excluded embedding parameters
        - Chinchilla included all parameters
        Ref: https://arxiv.org/abs/2203.15556 (Chinchilla paper)
        Ref: https://arxiv.org/abs/2001.08361 (Kaplan et al. original scaling laws paper)

        Returns a dict with counts for each parameter group, so downstream analysis
        can experiment with which combination gives the cleanest scaling laws.

        Key distinction for looped architecture:
        - recur_block: parameters executed num_recur times (reused)
        - prelude/coda/inject: parameters executed once per forward pass
        """
        # Count each group separately
        wte = sum(p.numel() for p in self.transformer.wte.parameters())
        value_embeds = 0  # Not used in looped architecture
        lm_head = sum(p.numel() for p in self.lm_head.parameters())

        # Split transformer matrices by execution pattern
        prelude = sum(p.numel() for p in self.transformer.prelude.parameters())
        recur_block = sum(p.numel() for p in self.transformer.recur.parameters())
        coda = sum(p.numel() for p in self.transformer.coda.parameters())
        inject = sum(p.numel() for p in self.inject.parameters()) if self.config.input_injection != "passthrough" else 0

        # Scalars: RMSNorm scale parameters
        scalars = (
            sum(p.numel() for p in self.norm_emb.parameters())
            + sum(p.numel() for p in self.norm_recur.parameters())
            + sum(p.numel() for p in self.norm_final.parameters())
        )

        total = wte + value_embeds + lm_head + prelude + recur_block + coda + inject + scalars
        assert total == sum(p.numel() for p in self.parameters()), "Parameter count mismatch"

        return {
            'wte': wte,
            'value_embeds': value_embeds,
            'lm_head': lm_head,
            'prelude': prelude,
            'recur_block': recur_block,
            'coda': coda,
            'inject': inject,
            'scalars': scalars,
            'total': total,
        }

    def effective_params(self, num_recur=None):
        """
        Compute effective parameter count accounting for parameter reuse in recurrent block.

        The recurrent block's parameters are executed num_recur times, while prelude/coda
        run once. This gives a measure of "parameter uses" rather than unique parameters.

        Args:
            num_recur: Number of recurrences to assume. If None, uses train_recur_mean.

        Returns:
            Effective parameter count (weighted by usage)
        """
        if num_recur is None:
            num_recur = int(self.config.train_recur_mean)

        counts = self.num_scaling_params()

        # Parameters that run once per forward pass
        once = counts['wte'] + counts['lm_head'] + counts['prelude'] + counts['coda'] + counts['scalars']

        # Parameters that run num_recur times per forward pass
        reused = counts['recur_block'] + counts['inject']

        effective = once + (reused * num_recur)
        return effective

    def _state_transfer(self, e, s=None, warm_start_state=None):
        """
        Handle state initialization and input injection based on input_injection mode.

        Combines state initialization and input injection into a single operation:
        - For "inject_init_prelude" and "inject_init_random": returns inject(concat(e, s))
        - For "passthrough": returns s directly (no injection layer)

        Args:
            e: Prelude output (B, T, n_embd)
            s: Current recurrent state (B, T, n_embd), or None for initial recurrence
            warm_start_state: Optional warm-start state from previous token (B, 1, n_embd) or (B, T, n_embd)

        Returns:
            Input to the recurrent block u (B, T, n_embd)
        """
        T = e.size(1)

        # Initialize state on first recurrence (when s is None)
        if s is None:
            if warm_start_state is not None:
                # warm_start_state may be (B, 1, h) from last token - broadcast to match e's shape (B, T, h)
                if warm_start_state.size(1) != T:
                    s = warm_start_state.expand(-1, T, -1)
                else:
                    s = warm_start_state
            elif self.config.input_injection == "inject_init_random":
                # Sample initial state from N(0, 1/sqrt(d)) per Geiping et al. Section 3.3
                noise_std = self.config.n_embd ** -0.5
                s = torch.randn_like(e) * noise_std
            else:
                # Both "inject_init_prelude" and "passthrough" start with prelude output
                s = e

        # Apply injection or passthrough based on mode
        if self.config.input_injection == "passthrough":
            # Pure recurrence: just pass state through directly
            return s
        else:
            # "inject_init_prelude" or "inject_init_random": use injection layer
            return self.inject(torch.cat([e, s], dim=-1))

    def setup_optimizer(
        self,
        unembedding_lr=0.004,
        embedding_lr=0.2,
        matrix_lr=0.02,
        weight_decay=0.0,
        adam_betas=(0.8, 0.95),
    ):
        """
        Create combined MuonAdamW optimizer: Muon for 2D matrix params, AdamW for embeddings.
        """
        model_dim = self.config.n_embd
        ddp, rank, local_rank, world_size = get_dist_info()

        # Separate out all parameters into groups
        # Collect all RMSNorm parameters (1D scale weights)
        norm_params = list(self.norm_emb.parameters()) + list(self.norm_recur.parameters()) + list(self.norm_final.parameters())
        for block in list(self.transformer.prelude) + list(self.transformer.recur) + list(self.transformer.coda):
            norm_params += list(block.n1.parameters()) + list(block.n2.parameters())
            norm_params += list(block.n3.parameters()) + list(block.n4.parameters())
        norm_param_set = set(norm_params)

        # Matrix params: all block params except norms (include inject only if not passthrough)
        inject_params_list = list(self.inject.parameters()) if self.config.input_injection != "passthrough" else []
        matrix_params = [
            p
            for p in (
                list(self.transformer.prelude.parameters())
                + list(self.transformer.recur.parameters())
                + list(self.transformer.coda.parameters())
                + inject_params_list
            )
            if p not in norm_param_set
        ]
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        assert len(list(self.parameters())) == len(matrix_params) + len(embedding_params) + len(lm_head_params) + len(norm_params)

        # Scale the LR for the AdamW parameters by ∝1/√dmodel (tuned for 768 dim model)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print0(f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")

        # Build param_groups with all required fields explicit
        param_groups = [
            # AdamW groups (embeddings, lm_head, norms)
            dict(kind="adamw", params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind="adamw", params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind="adamw", params=norm_params, lr=0.005 * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
        ]
        # Muon groups (matrix params, grouped by shape for stacking)
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(
                dict(
                    kind="muon",
                    params=group_params,
                    lr=matrix_lr,
                    momentum=0.95,
                    ns_steps=5,
                    beta2=0.95,
                    weight_decay=weight_decay,
                )
            )

        Factory = DistMuonAdamW if ddp else MuonAdamW
        optimizer = Factory(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def _predict(self, s, cos_sin, kv_cache=None, kv_budget=None):
        """
        Run coda blocks + norm_final + lm_head to produce logits from recurrent state.

        This is the final prediction stage: coda attention → final norm → lm_head → softcap.
        Coda KV cache entries are written in-place, so calling this multiple times with the
        same kv_cache (before advance()) simply overwrites the same positions — safe for
        collecting intermediate predictions at each recurrence step.

        Args:
            s: Recurrent state after norm_recur, shape (B, T, n_embd)
            cos_sin: Rotary embedding tuple for current positions
            kv_cache: Optional KV cache (entries overwritten in-place)
            kv_budget: KV budget (needed for layer index computation)

        Returns:
            logits: (B, T, vocab_size) in float32, with logit softcap applied
        """
        x = s
        for i, block in enumerate(self.transformer.coda):
            layer_idx = self._get_kv_layer_idx("coda", i, kv_budget)
            x = block(x, cos_sin, self.window_sizes[self.config.n_prelude + self.config.n_recur_block + i], kv_cache, layer_idx)
        x = self.norm_final(x)

        softcap = self.config.logit_softcap
        logits = self.lm_head(x)  # (B, T, padded_vocab_size)
        logits = logits[..., : self.config.vocab_size]  # slice to remove padding
        logits = logits.float()  # fp32 for softcap and loss computation
        logits = softcap * torch.tanh(logits / softcap)
        return logits

    def forward(
        self,
        idx,
        targets=None,
        kv_cache=None,
        loss_reduction="mean",
        num_recur=None,
        warm_start_state=None,
        return_intermediate_logits: bool = False,
        return_intermediate_states: bool = False,
    ):
        B, T = idx.size()
        if num_recur is None:
            num_recur = int(self.config.train_recur_mean)

        kv_budget = kv_cache.kv_budget if kv_cache is not None else None

        # Grab the rotary embeddings for the current sequence length (they are of shape (1, seq_len, 1, head_dim/2))
        assert self.cos.size(1) >= T, f"Sequence length grew beyond the rotary embeddings cache: {T} > {self.cos.size(1)}"
        assert idx.device == self.cos.device, f"Rotary embeddings and idx are on different devices: {idx.device} != {self.cos.device}"
        assert self.cos.dtype == torch.bfloat16, "Rotary embeddings must be in bfloat16"
        # if kv cache exists, we need to offset the rotary embeddings to the current position in the cache
        if kv_cache is not None and T == 1 and not torch.all(kv_cache.cache_seqlens == kv_cache.cache_seqlens[0]):
            # Per-element rotary positions for batched decode with non-uniform prompt lengths
            # self.cos is (1, max_seq, 1, hd/2), index with (B,) positions → (B, 1, hd/2) → unsqueeze → (B, 1, 1, hd/2)
            # This broadcasts correctly with q/k shapes (B, T=1, H, hd)
            positions = kv_cache.cache_seqlens.long()
            cos_sin = (
                self.cos[0, positions].unsqueeze(1),
                self.sin[0, positions].unsqueeze(1),
            )
        else:
            T0 = 0 if kv_cache is None else kv_cache.get_pos()
            cos_sin = (
                self.cos[:, T0 : T0 + T],
                self.sin[:, T0 : T0 + T],
            )  # truncate cache to current sequence length

        # 1. Embedding + norm
        x = self.transformer.wte(idx)
        x = self.norm_emb(x)

        # 2. Prelude blocks (run once)
        for i, block in enumerate(self.transformer.prelude):
            layer_idx = self._get_kv_layer_idx("prelude", i, kv_budget)
            x = block(x, cos_sin, self.window_sizes[i], kv_cache, layer_idx)
        e = x  # prelude output, used for injection into each recurrence

        # 3. Initialize state variable
        s = None
        # Collect intermediate logits when requested.
        # Coda KV cache entries are overwritten in-place at the same positions (before advance()),
        # so the final recurrence leaves the correct cache state.
        intermediate_logits = [] if return_intermediate_logits else None
        intermediate_states = [] if return_intermediate_states else None

        # 4. Recurrent block (run num_recur times)
        for i in range(num_recur):
            # State transfer: handles initialization (on i==0) and input injection
            u = self._state_transfer(e, s=s, warm_start_state=warm_start_state)
            # Run recur blocks with KV cache
            for j, block in enumerate(self.transformer.recur):
                layer_idx = self._get_kv_layer_idx("recur", j, kv_budget, recur_iter=i)
                u = block(u, cos_sin, self.window_sizes[self.config.n_prelude + j], kv_cache, layer_idx)
            s = self.norm_recur(u)  # nc: rescale at end of recurrent block
            # Truncated BPTT: detach gradients for recurrences before the last bptt_k
            # This limits gradient flow depth to bptt_k * n_recur_block layers through recurrence
            if self.config.bptt_k is not None and i < num_recur - self.config.bptt_k:
                s = s.detach()

            if return_intermediate_states:
                intermediate_states.append(s)
            if return_intermediate_logits:
                intermediate_logits.append(self._predict(s, cos_sin, kv_cache, kv_budget))

        # 5. Coda + norm_final + lm_head → logits (final prediction)
        # When collecting intermediate logits, the last entry is already the final prediction
        # (and the coda KV cache already has the correct final entries from that call).
        logits = intermediate_logits[-1] if return_intermediate_logits else self._predict(s, cos_sin, kv_cache, kv_budget)

        # Advance KV cache position after all layers are done
        if kv_cache is not None:
            kv_cache.advance(T)

        if targets is not None:
            # training: given the targets, compute and return the loss
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
                reduction=loss_reduction,
            )
            return loss
        elif return_intermediate_states:
            return logits, s, intermediate_logits, intermediate_states
        elif return_intermediate_logits:
            return logits, s, intermediate_logits
        else:
            return logits, s

    def forward_gated(
        self,
        idx: torch.Tensor,
        gate_config: GateConfig,
        num_recur: int | None = None,
        kv_cache=None,
        warm_start_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, GateStats]:
        """
        Forward pass with early-exit gating.

        Like forward(), but each (B, T) position can exit the recurrence loop
        independently when the gate condition fires. Exited positions have their
        state frozen; the loop breaks early when all positions have exited.

        Only kv_budget=1 is supported. KV cache entries for exited tokens are
        preserved from their exit iteration (not overwritten by later iterations).

        Args:
            idx: input token ids, shape (B, T)
            gate_config: GateConfig specifying gate type, threshold, etc.
            num_recur: max recurrence depth (default: train_recur_mean)
            kv_cache: optional KVCache (must have kv_budget=1 if provided)
            warm_start_state: optional (B, 1, D) state from previous token

        Returns:
            logits: (B, T, vocab_size) in float32
            s: (B, T, n_embd) final gated recurrent state
            stats: GateStats with per-position exit depths
        """
        B, T = idx.size()
        if num_recur is None:
            num_recur = int(self.config.train_recur_mean)

        kv_budget = kv_cache.kv_budget if kv_cache is not None else None
        if kv_budget is not None:
            assert kv_budget == 1, "forward_gated only supports kv_budget=1"

        # Rotary embeddings (same logic as forward)
        assert self.cos.size(1) >= T
        if kv_cache is not None and T == 1 and not torch.all(kv_cache.cache_seqlens == kv_cache.cache_seqlens[0]):
            positions = kv_cache.cache_seqlens.long()
            cos_sin = (self.cos[0, positions].unsqueeze(1), self.sin[0, positions].unsqueeze(1))
        else:
            T0 = 0 if kv_cache is None else kv_cache.get_pos()
            cos_sin = (self.cos[:, T0:T0 + T], self.sin[:, T0:T0 + T])

        # 1. Embedding + prelude (identical to forward)
        x = self.transformer.wte(idx)
        x = self.norm_emb(x)
        for i, block in enumerate(self.transformer.prelude):
            layer_idx = self._get_kv_layer_idx("prelude", i, kv_budget)
            x = block(x, cos_sin, self.window_sizes[i], kv_cache, layer_idx)
        e = x

        # 2. Gated recurrence
        gate = GateChecker(gate_config, B, T, device=e.device)
        needs_logits = gate_config.gate_type in ("kl_divergence", "entropy")
        s = None

        # KV cache save buffers: preserve recur KV entries for exited tokens
        kv_save_k: list[torch.Tensor] | None = None
        kv_save_v: list[torch.Tensor] | None = None
        recur_layer_indices: list[int] | None = None
        if kv_cache is not None:
            recur_layer_indices = [
                self._get_kv_layer_idx("recur", j, 1, recur_iter=0)
                for j in range(self.config.n_recur_block)
            ]
            T_start = kv_cache.get_pos()
            # Allocate empty buffers — filled incrementally as tokens exit
            sample_k, _ = kv_cache.get_layer_cache(recur_layer_indices[0])
            kv_save_k = [torch.empty(B, T, sample_k.shape[-2], sample_k.shape[-1],
                                     device=e.device, dtype=sample_k.dtype)
                         for _ in recur_layer_indices]
            kv_save_v = [torch.empty_like(kv_save_k[0]) for _ in recur_layer_indices]

        for i in range(num_recur):
            u = self._state_transfer(e, s=s, warm_start_state=warm_start_state)
            for j, block in enumerate(self.transformer.recur):
                layer_idx = self._get_kv_layer_idx("recur", j, 1, recur_iter=i)
                u = block(u, cos_sin, self.window_sizes[self.config.n_prelude + j], kv_cache, layer_idx)
            s_new = self.norm_recur(u)

            # Freeze state for already-exited tokens
            if s is not None and gate.exited.any():
                s_new = torch.where(gate.exited.unsqueeze(-1), s, s_new)

            # Gate check
            logits_i = None
            if needs_logits:
                logits_i = self._predict(s_new, cos_sin, kv_cache, kv_budget)
            newly_exited = gate.step(i, s_new, logits=logits_i)

            # Save recur KV entries for newly exited tokens
            if kv_cache is not None and newly_exited.any():
                for j_layer, li in enumerate(recur_layer_indices):
                    k_layer, v_layer = kv_cache.get_layer_cache(li)
                    # k_layer is (B, max_seq, H, D); we care about positions T_start:T_start+T
                    k_slice = k_layer[:, T_start:T_start + T]  # (B, T, H, D) view
                    v_slice = v_layer[:, T_start:T_start + T]
                    # Copy entries for newly exited positions
                    ne = newly_exited.unsqueeze(-1).unsqueeze(-1)  # (B, T, 1, 1)
                    kv_save_k[j_layer] = torch.where(ne, k_slice, kv_save_k[j_layer])
                    kv_save_v[j_layer] = torch.where(ne, v_slice, kv_save_v[j_layer])

            s = s_new
            if gate.exited.all():
                break

        # Restore saved KV entries for all tokens that exited early
        if kv_cache is not None and gate.exited.any():
            ex = gate.exited.unsqueeze(-1).unsqueeze(-1)  # (B, T, 1, 1)
            for j_layer, li in enumerate(recur_layer_indices):
                k_layer, v_layer = kv_cache.get_layer_cache(li)
                k_slice = k_layer[:, T_start:T_start + T]
                v_slice = v_layer[:, T_start:T_start + T]
                k_slice.copy_(torch.where(ex, kv_save_k[j_layer], k_slice))
                v_slice.copy_(torch.where(ex, kv_save_v[j_layer], v_slice))

        # 3. Final prediction (also writes correct coda KV)
        logits = self._predict(s, cos_sin, kv_cache, kv_budget)

        if kv_cache is not None:
            kv_cache.advance(T)

        stats = gate.finalize(num_recur)
        return logits, s, stats
