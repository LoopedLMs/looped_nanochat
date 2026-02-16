"""
Train a learned exit gate for looped transformers.

The model is frozen; only the gate (nn.Linear(d_model, 1)) is trained.
The gate learns per-token, per-depth exit probabilities that minimize
task loss while maximizing the entropy of the exit distribution (Ouro-style).

Training approach:
  1. Run frozen model forward to get intermediate states at each depth
  2. Gate produces exit probability p_i = sigmoid(W @ s_i + b) at each depth
  3. Soft mixture: exit-at-depth distribution e_i weights the per-depth losses
  4. Loss = CE_mix + beta * reg [+ optional lambda * E[depth] / num_recur]
     where reg = -H(weights) (entropy) or KL(weights || geometric(rate))

Usage:
    uv run python scripts/train_gate.py --model-tag r6_1.35e19_s20 --source sft
    torchrun --standalone --nproc_per_node=4 -m scripts.train_gate -- --model-tag r6_1.35e19_s20
"""

import argparse
import math
import os

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import time
from contextlib import nullcontext

import torch
import torch.distributed as dist
import torch.nn.functional as F
import wandb

from nanochat.checkpoint_manager import load_model
from nanochat.common import (
    DummyWandb,
    autodetect_device_type,
    compute_cleanup,
    compute_init,
    get_base_dir,
    print0,
)
from nanochat.gates import LearnedGate, compute_depth_weights, exit_distribution_entropy, geometric_target, kl_to_target
from nanochat.tokenizer import get_token_bytes, get_tokenizer
from tasks.common import TaskMixture
from tasks.customjson import CustomJSON
from tasks.gsm8k import GSM8K
from tasks.mmlu import MMLU
from tasks.smoltalk import SmolTalk
from tasks.spellingbee import SimpleSpelling, SpellingBee

# -----------------------------------------------------------------------------
# CLI arguments
parser = argparse.ArgumentParser(description="Train a learned exit gate")
# Logging
parser.add_argument("--run", type=str, default="dummy", help="wandb run name ('dummy' disables wandb)")
# Runtime
parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")
parser.add_argument("--dtype", type=str, default="bfloat16", help="float32|bfloat16")
# Model loading
parser.add_argument("--source", type=str, default="sft", choices=["base", "sft", "rl"], help="checkpoint source")
parser.add_argument("--model-tag", type=str, default=None, help="model tag to load")
parser.add_argument("--model-step", type=int, default=None, help="model step to load")
# Gate training
parser.add_argument("--num-recur", type=int, default=None, help="recurrence depth (default: model default, must be >= 2)")
parser.add_argument("--reg-type", type=str, default="entropy", choices=["entropy", "geometric"],
                    help="regularization type: 'entropy' = maximize H(weights) (uniform prior), "
                         "'geometric' = minimize KL(weights || geometric(rate))")
parser.add_argument("--reg-weight", "--entropy-weight", type=float, default=0.05, help="beta: regularization weight")
parser.add_argument("--geometric-rate", type=float, default=0.2, help="exit rate for geometric target (higher = earlier exits)")
parser.add_argument("--depth-weight", type=float, default=0.0, help="lambda: optional depth penalty weight (0 = disabled)")
parser.add_argument("--lr", type=float, default=1e-3, help="learning rate for gate (Adam)")
parser.add_argument("--gate-input", type=str, default="state", choices=["state", "state_delta"],
                    help="gate input mode: 'state' = s_i only, 'state_delta' = [s_i, s_i - s_{i-1}]")
# Training horizon
parser.add_argument("--num-iterations", type=int, default=-1, help="max training steps (-1 = full epoch)")
# Batch sizes
parser.add_argument("--max-seq-len", type=int, default=2048, help="max context length")
parser.add_argument("--device-batch-size", type=int, default=16, help="per-device batch size")
parser.add_argument("--total-batch-size", type=int, default=-1, help="total batch size in tokens (-1 = no gradient accumulation)")
# Evaluation
parser.add_argument("--eval-every", type=int, default=100, help="evaluate every N steps (-1 = disable)")
parser.add_argument("--eval-steps", type=int, default=10, help="number of val batches per evaluation")
# Output
parser.add_argument("--output-tag", type=str, default=None, help="output tag for saved gate (default: model-tag)")
args = parser.parse_args()
user_config = vars(args).copy()
# -----------------------------------------------------------------------------

# Compute init
device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0
ptdtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type == "cuda" else nullcontext()
synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None

# wandb
use_dummy_wandb = args.run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanochat-gate", name=args.run, config=user_config)

# Load model (frozen)
model, tokenizer, meta = load_model(args.source, device, phase="eval", model_tag=args.model_tag, step=args.model_step)
model.eval()
model.requires_grad_(False)

token_bytes = get_token_bytes(device=device)

num_recur = args.num_recur if args.num_recur is not None else int(model.config.train_recur_mean)
assert num_recur >= 2, f"num_recur must be >= 2 for gate gradients to flow (got {num_recur})"
print0(f"Model: {args.model_tag} (source={args.source}), num_recur={num_recur}")
print0(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

# Create gate (only trainable module)
gate = LearnedGate(model.config.n_embd, input_mode=args.gate_input).to(device)
gate_params = sum(p.numel() for p in gate.parameters())
print0(f"Gate params: {gate_params:,} (d_model={model.config.n_embd}, input_mode={args.gate_input})")

# Precompute regularization target
if args.reg_type == "geometric":
    reg_target = geometric_target(num_recur, args.geometric_rate, device)
    print0(f"Reg: KL to geometric(rate={args.geometric_rate}), target={reg_target.tolist()}")
else:
    reg_target = None
    print0(f"Reg: entropy (uniform prior)")

optimizer = torch.optim.Adam(gate.parameters(), lr=args.lr)

# Gradient accumulation
tokens_per_micro = args.device_batch_size * args.max_seq_len
if args.total_batch_size > 0:
    world_tokens_per_micro = tokens_per_micro * ddp_world_size
    assert args.total_batch_size % world_tokens_per_micro == 0, (
        f"total_batch_size ({args.total_batch_size}) must be divisible by "
        f"world_tokens_per_micro ({world_tokens_per_micro} = {args.device_batch_size} * {args.max_seq_len} * {ddp_world_size})"
    )
    grad_accum_steps = args.total_batch_size // world_tokens_per_micro
else:
    grad_accum_steps = 1
print0(f"Gradient accumulation steps: {grad_accum_steps}")

# ---------------------------------------------------------------------------
# SFT data (same mixture as chat_sft.py)

base_dir = get_base_dir()
identity_conversations_filepath = os.path.join(base_dir, "identity_conversations.jsonl")
train_dataset = TaskMixture(
    [
        SmolTalk(split="train"),
        MMLU(subset="auxiliary_train", split="train"),
        GSM8K(subset="main", split="train"),
        GSM8K(subset="main", split="train"),
        CustomJSON(filepath=identity_conversations_filepath),
        CustomJSON(filepath=identity_conversations_filepath),
        SimpleSpelling(size=200000, split="train"),
        SpellingBee(size=80000, split="train"),
    ]
)
val_dataset = TaskMixture(
    [
        SmolTalk(split="test"),
        MMLU(subset="all", split="test", stop=5200),
        GSM8K(subset="main", split="test", stop=420),
    ]
)


def sft_data_generator(dataset, split):
    """BOS-aligned bestfit-pad data generator (simplified from chat_sft.py)."""
    dataset_size = len(dataset)
    row_capacity = args.max_seq_len + 1
    bos_token = tokenizer.get_bos_token_id()

    conv_buffer = []
    cursor = ddp_rank
    consumed = ddp_rank
    buffer_size = 100

    def refill_buffer():
        nonlocal cursor
        while len(conv_buffer) < buffer_size:
            conversation = dataset[cursor]
            ids, _ = tokenizer.render_conversation(conversation)
            conv_buffer.append(ids)
            cursor += ddp_world_size
            if cursor >= dataset_size:
                cursor = cursor % dataset_size

    while True:
        rows = []
        row_lengths = []
        for _ in range(args.device_batch_size):
            row = []
            padded = False
            while len(row) < row_capacity:
                while len(conv_buffer) < buffer_size:
                    refill_buffer()

                remaining = row_capacity - len(row)
                best_idx = -1
                best_len = 0
                for i, conv in enumerate(conv_buffer):
                    conv_len = len(conv)
                    if conv_len <= remaining and conv_len > best_len:
                        best_idx = i
                        best_len = conv_len

                if best_idx >= 0:
                    conv = conv_buffer.pop(best_idx)
                    row.extend(conv)
                    consumed += ddp_world_size
                else:
                    content_len = len(row)
                    row.extend([bos_token] * remaining)
                    padded = True
                    break

            if padded:
                row_lengths.append(content_len)
            else:
                row_lengths.append(row_capacity)
            rows.append(row[:row_capacity])

        # Check epoch completion
        epoch_done = consumed >= dataset_size

        use_cuda = device_type == "cuda"
        batch_tensor = torch.tensor(rows, dtype=torch.long, pin_memory=use_cuda)
        inputs = batch_tensor[:, :-1].to(device=device, dtype=torch.int32, non_blocking=use_cuda)
        targets = batch_tensor[:, 1:].to(device=device, dtype=torch.int64, non_blocking=use_cuda)

        for i, content_len in enumerate(row_lengths):
            if content_len < row_capacity:
                targets[i, content_len - 1 :] = -1

        yield inputs, targets, epoch_done


# ---------------------------------------------------------------------------
# Gate training step


def gate_train_step(
    model, gate, inputs, targets, num_recur, reg_weight, depth_weight, autocast_ctx,
    token_bytes: torch.Tensor | None = None,
    reg_type: str = "entropy",
    reg_target: torch.Tensor | None = None,
):
    """
    One training step for the learned gate.

    Loss = CE_mix + beta * reg [+ lambda * E[depth] / num_recur]
      where reg = -H(weights) if reg_type="entropy" (maximize entropy)
                = KL(weights || target) if reg_type="geometric"

    Returns dict with loss components and stats for logging.
    """
    B, T = inputs.size()

    # 1. Frozen model forward — get intermediate states (no grad for model)
    with torch.no_grad(), autocast_ctx:
        _, _, _, intermediate_states = model(
            inputs, num_recur=num_recur,
            return_intermediate_states=True,
        )
    # intermediate_states: list of num_recur tensors, each (B, T, D) in bf16

    # 2. Gate probabilities (WITH gradient through gate)
    exit_probs = []
    for i, s_i in enumerate(intermediate_states):
        s_prev = intermediate_states[i - 1].float() if i > 0 else None
        exit_probs.append(gate(s_i.float(), s_prev))

    # 3. Compute depth weights (exit-at-depth distribution)
    weights = compute_depth_weights(exit_probs)

    # 4. Task loss: sequential per-depth CE weighted by exit probability
    # Process one depth at a time to avoid storing all intermediate logits
    valid_mask = targets != -1
    valid_mask_f = valid_mask.to(torch.float32)
    num_valid = valid_mask_f.sum().clamp(min=1)
    cos_sin = (model.cos[:, :T], model.sin[:, :T])

    # Precompute byte lengths for bpb metric
    task_bpb = None
    if token_bytes is not None:
        targets_safe = torch.where(valid_mask, targets, torch.zeros_like(targets))
        byte_lengths = token_bytes[targets_safe]
        byte_mask = (byte_lengths > 0) & valid_mask
        byte_mask_f = byte_mask.to(torch.float32)
        total_bytes = byte_lengths.sum().clamp(min=1)

    task_loss = torch.zeros(1, device=inputs.device)
    task_nats = torch.zeros(1, device=inputs.device) if token_bytes is not None else None
    for i in range(num_recur):
        with torch.no_grad(), autocast_ctx:
            logits_i = model._predict(intermediate_states[i], cos_sin)
        per_token_loss = F.cross_entropy(
            logits_i.view(-1, logits_i.size(-1)),
            targets.view(-1),
            ignore_index=-1,
            reduction="none",
        ).view(B, T)
        # Weight by exit probability at this depth (differentiable through gate)
        task_loss = task_loss + (weights[i] * per_token_loss).sum() / num_valid
        if task_nats is not None:
            task_nats = task_nats + (weights[i] * per_token_loss * byte_mask_f).sum()

    if task_nats is not None:
        task_bpb = (task_nats / (math.log(2) * total_bytes)).item()

    # 5. Regularization
    entropy = exit_distribution_entropy(weights)
    entropy_mean = (entropy * valid_mask_f).sum() / num_valid

    if reg_type == "geometric" and reg_target is not None:
        # KL(weights || geometric_target): shapes distribution toward early exit
        kl = kl_to_target(weights, reg_target)
        kl_mean = (kl * valid_mask_f).sum() / num_valid
        reg_loss = reg_weight * kl_mean
    else:
        # Entropy: maximize H(weights) → uniform prior
        kl_mean = None
        reg_loss = -reg_weight * entropy_mean

    # 6. Optional depth penalty: E[depth] / num_recur
    expected_depth = sum((i + 1) * w.mean() for i, w in enumerate(weights))
    depth_penalty = expected_depth / num_recur

    loss = task_loss + reg_loss + depth_weight * depth_penalty

    # Stats for logging
    with torch.no_grad():
        mean_depth = expected_depth.item()
        # Std of exit depth from soft weights: sqrt(E[d^2] - E[d]^2)
        expected_depth_sq = sum((i + 1) ** 2 * w.mean() for i, w in enumerate(weights))
        var_depth = expected_depth_sq.item() - mean_depth ** 2
        depth_std = max(var_depth, 0.0) ** 0.5
        # Mean exit weight per depth (soft histogram)
        depth_weights = [w.mean().item() for w in weights]

    return {
        "loss": loss,
        "task_loss": task_loss.item(),
        "task_bpb": task_bpb,
        "entropy": entropy_mean.item(),
        "kl": kl_mean.item() if kl_mean is not None else 0.0,
        "depth_penalty": depth_penalty.item(),
        "expected_depth": mean_depth,
        "depth_std": depth_std,
        "depth_weights": depth_weights,
    }


# ---------------------------------------------------------------------------
# Evaluation


@torch.no_grad()
def evaluate_gate(model, gate, val_loader, num_recur, reg_weight, depth_weight, autocast_ctx, steps,
                  token_bytes=None, reg_type="entropy", reg_target=None):
    """Evaluate gate on validation data."""
    gate.eval()
    scalar_keys = ["task_loss", "task_bpb", "entropy", "kl", "expected_depth", "depth_std"]
    totals = {k: 0.0 for k in scalar_keys}
    depth_weights_sum: list[float] | None = None
    n = 0

    for i, (inputs, targets, _) in enumerate(val_loader):
        if i >= steps:
            break
        result = gate_train_step(model, gate, inputs, targets, num_recur, reg_weight, depth_weight, autocast_ctx,
                                 token_bytes=token_bytes, reg_type=reg_type, reg_target=reg_target)
        for k in scalar_keys:
            totals[k] += result[k]
        dw = result["depth_weights"]
        if depth_weights_sum is None:
            depth_weights_sum = list(dw)
        else:
            for j in range(len(dw)):
                depth_weights_sum[j] += dw[j]
        n += 1

    gate.train()
    out = {f"val/{k}": v / n for k, v in totals.items()}
    if depth_weights_sum is not None:
        out["val/depth_weights"] = [w / n for w in depth_weights_sum]
    return out


# ---------------------------------------------------------------------------
# Training loop

train_loader = sft_data_generator(train_dataset, "train")
val_loader_fn = lambda: sft_data_generator(val_dataset, "val")

smooth_loss = 0.0
ema_beta = 0.9
step = 0
last_step = False

print0(f"\nStarting gate training: reg_weight={args.reg_weight}, depth_weight={args.depth_weight}, lr={args.lr}")
print0(f"  device_batch_size={args.device_batch_size}, max_seq_len={args.max_seq_len}")
print0(f"  grad_accum_steps={grad_accum_steps}, ddp_world_size={ddp_world_size}")
print0(f"  eval_every={args.eval_every}, eval_steps={args.eval_steps}")

while True:
    # Evaluation
    if last_step or (args.eval_every > 0 and step % args.eval_every == 0):
        val_loader = val_loader_fn()
        val_stats = evaluate_gate(
            model, gate, val_loader, num_recur, args.reg_weight, args.depth_weight, autocast_ctx, args.eval_steps,
            token_bytes=token_bytes, reg_type=args.reg_type, reg_target=reg_target,
        )
        print0(
            f"Step {step:05d} | val_bpb={val_stats['val/task_bpb']:.4f} val_loss={val_stats['val/task_loss']:.4f} "
            f"H={val_stats['val/entropy']:.4f} KL={val_stats['val/kl']:.4f} "
            f"E[depth]={val_stats['val/expected_depth']:.2f} "
            f"std={val_stats['val/depth_std']:.2f}"
        )
        val_log = {"step": step}
        val_dw = val_stats.pop("val/depth_weights", None)
        val_log.update(val_stats)
        if val_dw is not None:
            bin_edges = [d + 0.5 for d in range(len(val_dw) + 1)]
            val_log["val/depth_hist"] = wandb.Histogram(np_histogram=(val_dw, bin_edges))
        wandb_run.log(val_log)

    # Save gate at end
    if master_process and last_step:
        output_tag = args.output_tag or args.model_tag or "gate"
        gate_dir = os.path.join(base_dir, "gate_checkpoints", output_tag)
        os.makedirs(gate_dir, exist_ok=True)
        gate_path = os.path.join(gate_dir, f"gate_beta{args.reg_weight:.2f}_step{step:06d}.pt")
        torch.save(
            {
                "gate_state_dict": gate.state_dict(),
                "config": user_config,
                "model_tag": args.model_tag,
                "source": args.source,
                "num_recur": num_recur,
                "reg_type": args.reg_type,
                "reg_weight": args.reg_weight,
                "geometric_rate": args.geometric_rate,
                "depth_weight": args.depth_weight,
                "step": step,
                "d_model": model.config.n_embd,
                "gate_input": args.gate_input,
            },
            gate_path,
        )
        print0(f"Saved gate to: {gate_path}")

    if last_step:
        break

    # --- Training step (with gradient accumulation) ---
    synchronize()
    t0 = time.time()

    gate.train()
    optimizer.zero_grad()
    accum_result = None
    epoch_done = False
    for _micro in range(grad_accum_steps):
        inputs, targets, micro_epoch_done = next(train_loader)
        epoch_done = epoch_done or micro_epoch_done
        result = gate_train_step(
            model, gate, inputs, targets, num_recur, args.reg_weight, args.depth_weight, autocast_ctx,
            token_bytes=token_bytes, reg_type=args.reg_type, reg_target=reg_target,
        )
        (result["loss"] / grad_accum_steps).backward()
        accum_result = result  # keep last micro-step stats for logging

    # DDP: sync gate gradients across ranks
    if ddp:
        for p in gate.parameters():
            if p.grad is not None:
                dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)

    optimizer.step()

    synchronize()
    t1 = time.time()
    dt = t1 - t0

    step += 1

    # Check stopping conditions
    if epoch_done:
        last_step = True
    if 0 < args.num_iterations <= step:
        last_step = True

    # Synchronize last_step across ranks
    if ddp:
        last_step_tensor = torch.tensor(last_step, dtype=torch.int32, device=device)
        dist.all_reduce(last_step_tensor, op=dist.ReduceOp.MAX)
        last_step = bool(last_step_tensor.item())

    # Logging
    loss_f = accum_result["loss"].item()
    smooth_loss = ema_beta * smooth_loss + (1 - ema_beta) * loss_f
    debiased_loss = smooth_loss / (1 - ema_beta ** step)

    if step % 10 == 0 or step == 1:
        print0(
            f"step {step:05d} | loss={debiased_loss:.4f} "
            f"(bpb={accum_result['task_bpb']:.4f} H={accum_result['entropy']:.4f} KL={accum_result['kl']:.4f} depth={accum_result['depth_penalty']:.4f}) | "
            f"E[d]={accum_result['expected_depth']:.2f} "
            f"std={accum_result['depth_std']:.2f} | "
            f"dt={dt * 1000:.0f}ms"
        )
    if step % 10 == 0:
        log_dict = {
            "step": step,
            "train/loss": debiased_loss,
            "train/task_loss": accum_result["task_loss"],
            "train/task_bpb": accum_result["task_bpb"],
            "train/entropy": accum_result["entropy"],
            "train/kl": accum_result["kl"],
            "train/depth_penalty": accum_result["depth_penalty"],
            "train/expected_depth": accum_result["expected_depth"],
            "train/depth_std": accum_result["depth_std"],
            "train/dt": dt,
        }
        dw = accum_result["depth_weights"]
        bin_edges = [d + 0.5 for d in range(len(dw) + 1)]
        log_dict["train/depth_hist"] = wandb.Histogram(np_histogram=(dw, bin_edges))
        wandb_run.log(log_dict)

print0(f"\nTraining complete. Steps: {step}")
wandb_run.finish()
compute_cleanup()
