"""
SFT with latent channel warm-start training.

Continues SFT from a checkpoint with recurrent warm-start: the final recurrent
state from position t-1 initializes the recurrent loop at position t, simulating
autoregressive decoding with latent state carry-forward.

Two-pass training per micro-step:
  Pass 1: forward through embedding + prelude + recurrence (skips coda + lm_head)
          to collect final recurrent states at each position
  Pass 2: forward with shifted states as warm_start, compute loss

Only assistant response positions use warm_start (via response_mask), simulating:
  - Parallel prefill for user/system tokens (no warm_start)
  - Sequential decoding for assistant responses (with warm_start)

Conversations are packed using best-fit-pad packing (multiple conversations per
row). The response_mask naturally prevents cross-conversation state leakage: only
assistant content tokens have mask=1, so warm_start is never applied to BOS, user,
system, or padding tokens at conversation boundaries.

Approximation note: the warm_start states come from a "cold" forward pass (pass 1
without warm_start itself). In real autoregressive decoding, subsequent tokens would
receive "warm" states (computed with warm_start from the previous token). This means
training slightly differs from inference for the 2nd+ assistant token. In practice,
this is a reasonable approximation: the model learns to use warm_start states, and
the recurrence loop refines the state regardless of initialization quality.

Run as:
python -m scripts.chat_sft_latent

Or torchrun for training:
torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft_latent -- --device-batch-size=16
"""

import argparse
import os

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import time
from contextlib import nullcontext
from dataclasses import asdict

import torch
import torch.distributed as dist
import wandb

from nanochat.checkpoint_manager import load_model, save_checkpoint
from nanochat.common import (
    DummyWandb,
    autodetect_device_type,
    compute_cleanup,
    compute_gradient_stats,
    compute_init,
    get_base_dir,
    get_num_recur_for_microstep,
    print0,
    sample_num_recurs_for_step,
)
from nanochat.loss_eval import evaluate_bpb
from nanochat.tokenizer import get_token_bytes
from tasks.common import TaskMixture
from tasks.customjson import CustomJSON
from tasks.gsm8k import GSM8K
from tasks.mmlu import MMLU
from tasks.smoltalk import SmolTalk
from tasks.spellingbee import SimpleSpelling, SpellingBee

# -----------------------------------------------------------------------------
# CLI arguments
parser = argparse.ArgumentParser(description="SFT with latent channel warm-start")
# Logging
parser.add_argument("--run", type=str, default="dummy", help="wandb run name ('dummy' disables wandb logging)")
# Runtime
parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")
parser.add_argument("--dtype", type=str, default="bfloat16", help="float32|bfloat16")
# Model loading
parser.add_argument("--init-source", type=str, default="sft", choices=["base", "sft", "sft_latent", "rl"],
                    help="checkpoint source to initialize from (default: sft)")
parser.add_argument("--model-tag", type=str, default=None, help="model tag to load from")
parser.add_argument("--model-step", type=int, default=None, help="model step to load from")
parser.add_argument("--output-tag", type=str, default=None, help="model tag to save to (defaults to model-tag)")
# Training horizon
parser.add_argument("--num-iterations", type=int, default=-1, help="number of optimization steps (-1 = full epoch)")
# Batch sizes
parser.add_argument("--max-seq-len", type=int, default=2048, help="max context length")
parser.add_argument("--device-batch-size", type=int, default=32, help="per-device batch size")
parser.add_argument("--total-batch-size", type=int, default=524288, help="total batch size in tokens")
# Optimization
parser.add_argument("--embedding-lr", type=float, default=0.2, help="learning rate for embedding parameters (Adam)")
parser.add_argument("--unembedding-lr", type=float, default=0.004, help="learning rate for unembedding parameters (Adam)")
parser.add_argument("--matrix-lr", type=float, default=0.02, help="learning rate for matrix parameters (Muon)")
parser.add_argument("--weight-decay", type=float, default=0.0, help="weight decay for embedding/unembedding parameters (Adam)")
parser.add_argument("--init-lr-frac", type=float, default=1.0, help="initial LR as fraction of base LR")
# Evaluation
parser.add_argument("--eval-every", type=int, default=150, help="evaluate val bpb every N steps (-1 = disable)")
parser.add_argument("--eval-tokens", type=int, default=20 * 524288, help="number of tokens to evaluate val loss on")
# Recurrence
parser.add_argument(
    "--recur-samples-per-step",
    type=int,
    default=8,
    help="Number of different num_recur values per gradient accumulation step (None = fixed, use model default)",
)
# Latent warm-start options
parser.add_argument(
    "--no-detach-warmup",
    action="store_true",
    help=(
        "Keep gradient graph for warm_start states (pass 1 runs with gradients). "
        "Allows the model to learn to produce better warm_start states, but roughly "
        "doubles activation memory. Consider reducing --device-batch-size. "
        "Default: detached (pass 1 uses torch.no_grad)."
    ),
)
# Output
parser.add_argument("--dry-run", action="store_true", help="log to wandb but skip checkpoints/report")
# Gradient tracking
parser.add_argument(
    "--track-gradients",
    type=str,
    choices=["none", "basic", "detailed"],
    default="basic",
    help="Gradient tracking level: none (disabled), basic (global norm), detailed (per-component norms)",
)
args = parser.parse_args()
user_config = vars(args).copy()
detach_warmup = not args.no_detach_warmup
# -----------------------------------------------------------------------------

# Compute init
device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0
ptdtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type == "cuda" else nullcontext()
synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None
get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0

# wandb logging init
use_dummy_wandb = args.run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanochat-sft", name=args.run, config=user_config)

# Load model checkpoint
model, tokenizer, meta = load_model(args.init_source, device, phase="train", model_tag=args.model_tag, step=args.model_step)
# Always compile blocks individually (never whole model) because the two-pass
# approach calls the model with different forward signatures per micro-step:
# pass 1 uses collect_recurrent_states (no lm_head), pass 2 returns loss.
# torch.compile(model) would trigger recompilation between these signatures.
orig_model = model
model.compile_blocks()
size = model.config.size
num_flops_per_token = model.estimate_flops()
tokens_per_fwdbwd = args.device_batch_size * args.max_seq_len  # tokens per iteration for a single rank
world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size  # total tokens per iteration for all ranks
assert args.total_batch_size % world_tokens_per_fwdbwd == 0
grad_accum_steps = args.total_batch_size // world_tokens_per_fwdbwd
print0(f"Tokens / micro-batch / rank: {args.device_batch_size} x {args.max_seq_len} = {tokens_per_fwdbwd:,}")
print0(f"Tokens / micro-batch: {world_tokens_per_fwdbwd:,}")
print0(f"Total batch size {args.total_batch_size:,} => gradient accumulation steps: {grad_accum_steps}")
print0(f"Latent warm-start: detach_warmup={detach_warmup}")
# Validate recur-samples-per-step (global across all ranks)
if args.recur_samples_per_step:
    total_micro_steps = ddp_world_size * grad_accum_steps
    if args.recur_samples_per_step > total_micro_steps:
        raise ValueError(
            f"--recur-samples-per-step ({args.recur_samples_per_step}) cannot exceed total micro-steps ({total_micro_steps} = {ddp_world_size} ranks * {grad_accum_steps} steps). "
            f"Decrease --recur-samples-per-step or --device-batch-size."
        )
    if total_micro_steps % args.recur_samples_per_step != 0:
        raise ValueError(
            f"Total micro-steps ({total_micro_steps} = {ddp_world_size} ranks * {grad_accum_steps} steps) must be evenly divisible by --recur-samples-per-step ({args.recur_samples_per_step}). "
            f"Adjust --device-batch-size to change grad_accum_steps."
        )
    microsteps_per_sample = total_micro_steps // args.recur_samples_per_step
    print0(f"Recurrence sampling: {args.recur_samples_per_step} global samples per step ({microsteps_per_sample} microsteps each across {ddp_world_size} ranks)")
else:
    print0(f"Recurrence sampling: fixed (using model default)")
token_bytes = get_token_bytes(device=device)

# Initialize the Optimizer
optimizer = model.setup_optimizer(
    unembedding_lr=args.unembedding_lr,
    embedding_lr=args.embedding_lr,
    matrix_lr=args.matrix_lr,
    weight_decay=args.weight_decay,
)
# Set the initial learning rate as a fraction of the base learning rate
for group in optimizer.param_groups:
    group["lr"] = group["lr"] * args.init_lr_frac
    group["initial_lr"] = group["lr"]

# SFT data mixture (same as chat_sft.py)
base_dir = get_base_dir()
identity_conversations_filepath = os.path.join(base_dir, "identity_conversations.jsonl")
train_dataset = TaskMixture(
    [
        SmolTalk(split="train"),  # 460K rows of general conversations
        MMLU(subset="auxiliary_train", split="train"),  # 100K rows of multiple choice problems
        GSM8K(subset="main", split="train"),  # 8K rows teaching simple math and (calculator) tool use
        GSM8K(subset="main", split="train"),  # 2 epochs of GSM8K
        CustomJSON(filepath=identity_conversations_filepath),  # 1000 rows of synthetic identity conversations
        CustomJSON(filepath=identity_conversations_filepath),  # 2 epochs of these
        SimpleSpelling(size=200000, split="train"),  # 200K rows of Simple Spelling
        SpellingBee(size=80000, split="train"),  # 80K rows of Spelling Bee
    ]
)  # total: ~856K rows
val_dataset = TaskMixture(
    [
        SmolTalk(split="test"),  # 24K rows in test set
        MMLU(subset="all", split="test", stop=5200),  # 5.2K to match the train ratios
        GSM8K(subset="main", split="test", stop=420),  # 420 to match the train ratios
    ]
)  # total: ~30K rows
last_step = False
approx_progress = 0.0
current_epoch = 1


# ---------------------------------------------------------------------------
# Helper: collect recurrent states (skip coda + lm_head to save memory)
# ---------------------------------------------------------------------------


def collect_recurrent_states(model, inputs, num_recur):
    """
    Run embedding -> prelude -> recurrence loop, skipping coda and lm_head.

    Returns the final recurrent state s of shape (B, T, n_embd).

    Mirrors the first 4 stages of GPT.forward() but omits the prediction head
    to avoid allocating the large (B, T, vocab_size) logits tensor.

    Caller controls the gradient context:
    - torch.no_grad(): states are detached (memory-efficient, default)
    - no context: states track gradients, BPTT truncation applied (--no-detach-warmup)
    """
    if num_recur is None:
        num_recur = int(model.config.train_recur_mean)

    B, T = inputs.size()
    cos_sin = (model.cos[:, :T], model.sin[:, :T])

    # 1. Embedding + norm
    x = model.transformer.wte(inputs)
    x = model.norm_emb(x)

    # 2. Prelude blocks
    for i, block in enumerate(model.transformer.prelude):
        x = block(x, cos_sin, model.window_sizes[i], None)
    e = x

    # 3. Recurrent block (num_recur iterations)
    s = None
    for i in range(num_recur):
        u = model._state_transfer(e, s=s)
        for j, block in enumerate(model.transformer.recur):
            u = block(u, cos_sin, model.window_sizes[model.config.n_prelude + j], None)
        s = model.norm_recur(u)
        # BPTT truncation when gradients are tracked (matches GPT.forward behavior)
        if s.requires_grad and model.config.bptt_k is not None and i < num_recur - model.config.bptt_k:
            s = s.detach()

    return s


# ---------------------------------------------------------------------------
# Data generator: bestfit-pad packing with response_mask for warm-start
# ---------------------------------------------------------------------------


def sft_latent_data_generator(split, buffer_size=100):
    """
    BOS-aligned packed dataloader for SFT with latent warm-start.

    Packs multiple conversations per row using best-fit algorithm (same as
    chat_sft.py), but also tracks the response_mask from render_conversation().
    Returns (inputs, targets, response_mask) where response_mask (B, T) is a
    bool tensor indicating assistant response positions for selective warm-start.

    The response_mask naturally prevents cross-conversation state leakage:
    non-assistant tokens (BOS, user, system, special tokens, padding) have
    mask=0, so warm_start is only applied to assistant content positions.
    """
    global last_step, approx_progress, current_epoch
    assert split in {"train", "val"}, "split must be 'train' or 'val'"
    dataset = train_dataset if split == "train" else val_dataset
    dataset_size = len(dataset)
    assert dataset_size > 0
    row_capacity = args.max_seq_len + 1  # +1 for target at last position
    bos_token = tokenizer.get_bos_token_id()

    # Conversation buffer: list of (ids, mask) tuples
    conv_buffer: list[tuple[list[int], list[int]]] = []
    cursor = ddp_rank  # Each rank processes different conversations
    consumed = ddp_rank
    epoch = 1
    it = 0

    def refill_buffer():
        nonlocal cursor, epoch
        while len(conv_buffer) < buffer_size:
            conversation = dataset[cursor]
            ids, mask = tokenizer.render_conversation(conversation)
            conv_buffer.append((ids, mask))
            cursor += ddp_world_size
            if cursor >= dataset_size:
                cursor = cursor % dataset_size
                epoch += 1

    while True:
        rows_ids = []
        rows_masks = []
        row_lengths = []

        for _ in range(args.device_batch_size):
            row_ids: list[int] = []
            row_mask: list[int] = []
            padded = False

            while len(row_ids) < row_capacity:
                while len(conv_buffer) < buffer_size:
                    refill_buffer()

                remaining = row_capacity - len(row_ids)

                # Find largest conversation that fits entirely
                best_idx = -1
                best_len = 0
                for i, (conv_ids, _) in enumerate(conv_buffer):
                    conv_len = len(conv_ids)
                    if conv_len <= remaining and conv_len > best_len:
                        best_idx = i
                        best_len = conv_len

                if best_idx >= 0:
                    conv_ids, conv_mask = conv_buffer.pop(best_idx)
                    row_ids.extend(conv_ids)
                    row_mask.extend(conv_mask)
                    consumed += ddp_world_size
                else:
                    # No conversation fits - pad the remainder
                    content_len = len(row_ids)
                    row_ids.extend([bos_token] * remaining)
                    row_mask.extend([0] * remaining)
                    padded = True
                    break

            if padded:
                row_lengths.append(content_len)
            else:
                row_lengths.append(row_capacity)
            rows_ids.append(row_ids[:row_capacity])
            rows_masks.append(row_mask[:row_capacity])

        # Stopping condition
        it += 1
        if 0 < args.num_iterations <= it and split == "train":
            last_step = True

        if split == "train":
            current_epoch = epoch
            if args.num_iterations > 0:
                approx_progress = it / args.num_iterations
            else:
                approx_progress = consumed / dataset_size
            if consumed >= dataset_size:
                last_step = True

        # Build tensors
        use_cuda = device_type == "cuda"
        batch_tensor = torch.tensor(rows_ids, dtype=torch.long, pin_memory=use_cuda)
        mask_tensor = torch.tensor(rows_masks, dtype=torch.long, pin_memory=use_cuda)
        inputs = batch_tensor[:, :-1].to(device=device, dtype=torch.int32, non_blocking=use_cuda)
        targets = batch_tensor[:, 1:].to(device=device, dtype=torch.int64, non_blocking=use_cuda)

        # Response mask: aligned with input positions (mask[:-1])
        # True for assistant response tokens, False for everything else
        response_mask = mask_tensor[:, :-1].to(device=device, dtype=torch.bool, non_blocking=use_cuda)

        # Mask out padding positions in targets (set to -1 = ignore_index)
        for i, content_len in enumerate(row_lengths):
            if content_len < row_capacity:
                targets[i, content_len - 1 :] = -1

        yield inputs, targets, response_mask


train_loader = sft_latent_data_generator("train")


def build_val_loader():
    """Wrap latent data generator to yield (x, y) for evaluate_bpb compatibility."""
    for x, y, _ in sft_latent_data_generator("val"):
        yield x, y


progress = 0  # will go from 0 to 1 over the course of the epoch


# Learning rate scheduler (warmdown: 80% constant, then linear decay)
def get_lr_multiplier(progress):
    # first 80% of training: no decay, then linearly ramp down to 0.
    return 1 if progress < 0.8 else 1 - (progress - 0.8) / 0.2


# Momentum scheduler for Muon optimizer
def get_muon_momentum(it):
    frac = min(it / 300, 1)
    momentum = (1 - frac) * 0.85 + frac * 0.95
    return momentum


# -----------------------------------------------------------------------------
# Training loop
x, y, response_mask = next(train_loader)  # prefetch the very first batch of data
min_val_bpb = float("inf")
smooth_train_loss = 0  # EMA of training loss
ema_beta = 0.9  # EMA decay factor
total_training_time = 0  # total wall-clock time of training
step = 0
while True:
    flops_so_far = num_flops_per_token * args.total_batch_size * step

    # Synchronize last_step across all ranks to avoid hangs in the distributed setting
    if ddp:
        last_step_tensor = torch.tensor(last_step, dtype=torch.int32, device=device)
        dist.all_reduce(last_step_tensor, op=dist.ReduceOp.MAX)
        last_step = bool(last_step_tensor.item())

    # once in a while: evaluate the val bpb (all ranks participate)
    # Evaluation is WITHOUT warm_start for comparability with regular SFT
    if last_step or (args.eval_every > 0 and step % args.eval_every == 0):
        model.eval()
        val_loader = build_val_loader()
        eval_steps = args.eval_tokens // (args.device_batch_size * args.max_seq_len * ddp_world_size)
        with autocast_ctx:
            val_bpb = evaluate_bpb(model, val_loader, eval_steps, token_bytes)
        print0(f"Step {step:05d} | Validation bpb: {val_bpb:.4f}")
        if val_bpb < min_val_bpb:
            min_val_bpb = val_bpb
        wandb_run.log(
            {
                "step": step,
                "total_training_flops": flops_so_far,
                "total_training_time": total_training_time,
                "val/bpb": val_bpb,
            }
        )
        model.train()

    # save checkpoint at the end of the run (only on master process)
    if master_process and last_step and not args.dry_run:
        output_dirname = args.output_tag or args.model_tag or f"s{size}"  # e.g. s12
        checkpoint_dir = os.path.join(base_dir, "chatsft_latent_checkpoints", output_dirname)
        save_checkpoint(
            checkpoint_dir,
            step,
            orig_model.state_dict(),
            None,  # note: we don't save optimizer state
            {
                "step": step,
                "val_bpb": val_bpb,  # loss at last step
                "model_config": asdict(model.config),
                "user_config": user_config,  # inputs to the training script
            },
        )

    if last_step:
        break

    # -------------------------------------------------------------------------
    # single training step
    # evaluate the gradient
    synchronize()
    t0 = time.time()

    # Pre-sample all num_recur values for this step (global across all ranks)
    sampled_num_recurs = sample_num_recurs_for_step(
        recur_samples_per_step=args.recur_samples_per_step,
        mean_recur=model.config.train_recur_mean,
        sigma=0.5,
        max_recur=model.config.train_recur_max,
        ddp=ddp,
        master_process=master_process,
        device=device,
    )

    for _micro_step in range(grad_accum_steps):
        # Get num_recur for this micro-step
        num_recur = get_num_recur_for_microstep(
            sampled_num_recurs=sampled_num_recurs,
            micro_step=_micro_step,
            ddp_rank=ddp_rank,
            ddp_world_size=ddp_world_size,
            grad_accum_steps=grad_accum_steps,
            recur_samples_per_step=args.recur_samples_per_step,
        )

        # Pass 1: collect final recurrent states at each position
        # Skips coda + lm_head to avoid allocating the (B, T, vocab_size) logits tensor.
        # When detach_warmup=True (default): no gradient through warm_start states.
        # When detach_warmup=False: gradient flows from pass 2 loss through warm_start
        #   back into pass 1 parameters, teaching the model to produce useful states.
        grad_context = torch.no_grad() if detach_warmup else nullcontext()
        with grad_context:
            with autocast_ctx:
                final_state = collect_recurrent_states(orig_model, x, num_recur)

        # Shift states by 1 position: state from position t-1 becomes warm_start for position t.
        # Position 0 gets zero-initialized (no previous state to carry forward).
        # torch.cat preserves grad graph when detach_warmup=False.
        B_cur, T_cur, D_cur = final_state.shape
        padding = final_state.new_zeros(B_cur, 1, D_cur)
        warm_start_state = torch.cat([padding, final_state[:, :-1, :]], dim=1)
        del final_state

        # Pass 2: forward with warm_start, compute loss on all non-padding tokens
        with autocast_ctx:
            loss = model(
                x,
                y,
                num_recur=num_recur,
                warm_start_state=warm_start_state,
                warm_start_mask=response_mask,
            )
        del warm_start_state
        train_loss = loss.detach()  # for logging
        loss = loss / grad_accum_steps  # each .backward() is a grad sum => normalize loss here
        loss.backward()
        x, y, response_mask = next(train_loader)  # prefetch the next batch while the GPU is busy
        progress = max(progress, approx_progress)  # only increase progress monotonically

    # Compute model health statistics: gradients and parameters (after all backward passes complete)
    model_health_stats = compute_gradient_stats(orig_model, args.track_gradients)

    # step the optimizer
    lrm = get_lr_multiplier(progress)
    muon_momentum = get_muon_momentum(step)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm
        if group.get("kind") == "muon":
            group["momentum"] = muon_momentum
    optimizer.step()
    model.zero_grad(set_to_none=True)

    # Extract effective learning rates from optimizer groups (for logging)
    effective_lr_unembed = optimizer.param_groups[0]["lr"]  # lm_head (AdamW)
    effective_lr_embed = optimizer.param_groups[1]["lr"]  # embedding (AdamW)
    effective_lr_muon = optimizer.param_groups[3]["lr"]  # first Muon group

    synchronize()
    t1 = time.time()
    dt = t1 - t0
    # -------------------------------------------------------------------------

    # State
    step += 1

    # logging
    train_loss_f = train_loss.item()  # raw unsmoothed loss from last microbatch
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f  # EMA the training loss
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta ** (step + 1))  # debias the EMA
    pct_done = 100 * progress
    tok_per_sec = int(args.total_batch_size / dt)
    # Note: MFU is approximate for two-pass training (actual FLOPs per token are ~1.7x
    # the estimate because pass 1 runs embedding + prelude + recurrence without coda/lm_head)
    flops_per_sec = num_flops_per_token * args.total_batch_size / dt
    promised_flops_per_sec_h100 = 989e12 * ddp_world_size  # bfloat16 H100 SXM
    mfu = 100 * flops_per_sec / promised_flops_per_sec_h100  # in %
    # For logging: report num_recur info
    if sampled_num_recurs is None:
        # Fixed: use model default
        logged_num_recur = int(model.config.train_recur_mean)
        logged_num_recur_str = f"{logged_num_recur}"
    else:
        # Sampled: report all values and mean
        logged_num_recur = sum(sampled_num_recurs) / len(sampled_num_recurs)
        logged_num_recur_str = f"{sampled_num_recurs} (mean={logged_num_recur:.1f})"
    if step > 10:
        total_training_time += dt  # only count the time after the first 10 steps
    print0(
        f"step {step:05d} ({pct_done:.2f}%) | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | dt: {dt * 1000:.2f}ms | tok/sec: {tok_per_sec:,} | mfu: {mfu:.2f} | epoch: {current_epoch} | num_recur: {logged_num_recur_str} | total time: {total_training_time / 60:.2f}m"
    )
    if step % 10 == 0:
        wandb_run.log(
            {
                "step": step,
                "total_training_flops": flops_so_far,
                "total_training_time": total_training_time,
                "train/loss": debiased_smooth_loss,
                "train/loss_raw": train_loss_f,  # raw unsmoothed loss from last microbatch
                "train/lrm": lrm,
                "train/dt": dt,
                "train/tok_per_sec": tok_per_sec,
                "train/mfu": mfu,
                "train/epoch": current_epoch,
                "train/num_recur": logged_num_recur,
                "lr/muon": effective_lr_muon,  # effective learning rate for Muon (matrix params)
                "lr/embed": effective_lr_embed,  # effective learning rate for embeddings
                "lr/unembed": effective_lr_unembed,  # effective learning rate for unembedding (lm_head)
                **{f"model_health/{k}": v for k, v in model_health_stats.items()},
            }
        )

# print a few more stats
print0(f"Peak memory usage: {get_max_memory() / 1024 / 1024:.2f}MiB")
print0(f"Total training time: {total_training_time / 60:.2f}m")
print0(f"Minimum validation bpb: {min_val_bpb:.4f}")

# Log to report
if not args.dry_run:
    from nanochat.report import get_report

    get_report().log(
        section="SFT Latent",
        data=[
            user_config,  # CLI args
            {  # stats about the training setup
                "Number of iterations": step,
                "DDP world size": ddp_world_size,
            },
            {  # stats about training outcomes
                "Minimum validation bpb": min_val_bpb,
            },
        ],
    )

# cleanup
wandb_run.finish()  # wandb run finish
compute_cleanup()
