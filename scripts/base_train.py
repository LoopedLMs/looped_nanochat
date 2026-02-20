"""
Train model. From root directory of the project, run as:

python -m scripts.base_train.py

or distributed as:

torchrun --nproc_per_node=8 -m scripts.base_train.py

If you are only on CPU/Macbook, you'll want to train a much much smaller LLM. Example:
python -m scripts.base_train --size=4 --max-seq-len=512 --device-batch-size=1 --eval-tokens=512 --core-metric-every=-1 --total-batch-size=512 --num-iterations=20
"""

import gc
import math
import os

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import argparse
import time
from contextlib import nullcontext

import torch
import wandb

from nanochat.checkpoint_manager import load_checkpoint, save_checkpoint
from nanochat.common import (
    DummyWandb,
    autodetect_device_type,
    compute_cleanup,
    compute_gradient_stats,
    compute_init,
    get_base_dir,
    get_num_recur_for_microstep,
    get_peak_flops,
    print0,
    print_banner,
    compute_cumulative_flops,
    get_scheduled_recur_mean,
    sample_num_recurs_for_step,
    sample_poisson_lognormal_recurrence,
    solve_iterations_for_target_flops,
)
from nanochat.dataloader import (
    tokenizing_distributed_data_loader_bos_bestfit,
    tokenizing_distributed_data_loader_with_state_bos_bestfit,
)
from nanochat.engine import Engine
from nanochat.flash_attention import HAS_FA2, HAS_FA3
from nanochat.gpt import GPT, GPTConfig
from nanochat.loss_eval import evaluate_bpb
from nanochat.report import get_report
from nanochat.tokenizer import get_token_bytes, get_tokenizer
from scripts.base_eval import evaluate_core

print_banner()

# -----------------------------------------------------------------------------
# CLI arguments
parser = argparse.ArgumentParser(description="Pretrain base model")
# Logging
parser.add_argument(
    "--run",
    type=str,
    default="dummy",
    help="wandb run name ('dummy' disables wandb logging)",
)
# Runtime
parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")
# Model architecture
parser.add_argument("--size", type=int, default=20, help="model size (model_dim = size * aspect_ratio)")
parser.add_argument("--aspect-ratio", type=int, default=64, help="model_dim = size * aspect_ratio")
parser.add_argument("--head-dim", type=int, default=128, help="target head dimension for attention")
parser.add_argument("--max-seq-len", type=int, default=2048, help="max context length")
parser.add_argument(
    "--window-pattern", type=str, default="LLSSSLLL", help="sliding window pattern tiled across layers: L=full, S=half context (e.g. 'SSL')"
)
# Looped Transformer config
parser.add_argument("--n-prelude", type=int, default=2, help="number of prelude layers")
parser.add_argument("--n-recur-block", type=int, default=4, help="number of layers in the recurrent block")
parser.add_argument("--n-coda", type=int, default=2, help="number of coda layers")
parser.add_argument("--train-recur-mean", type=float, default=4.0, help="mean recurrences during training (also default r at inference)")
parser.add_argument("--train-recur-max", type=int, default=16, help="max recurrences sampled during training")
parser.add_argument("--bptt-k", type=int, default=4, help="truncate backprop to last k recurrences (limits gradient depth)")
parser.add_argument(
    "--input-injection",
    type=str,
    default="inject_init_prelude",
    choices=["inject_init_prelude", "inject_init_random", "passthrough"],
    help="input injection mode: inject_init_prelude (default), inject_init_random, or passthrough (no injection)",
)
parser.add_argument(
    "--recur-samples-per-step",
    type=int,
    default=8,
    help="Number of different num_recur values per gradient accumulation step (0 = fixed, use model default)",
)
# Training horizon (only one used, in order of precedence)
parser.add_argument("--num-iterations", type=int, default=-1, help="explicit number of optimization steps (-1 = disable)")
parser.add_argument("--target-flops", type=float, default=-1.0, help="calculate num_iterations to reach target_flops (-1 = disable)")
parser.add_argument(
    "--target-param-data-ratio",
    type=int,
    default=7,
    help="calculate num_iterations to maintain data:param ratio (accounts for parameter reuse + slight overtrain), -1 = disable)",
)
# Optimization
parser.add_argument("--device-batch-size", type=int, default=32, help="per-device batch size")
parser.add_argument("--total-batch-size", type=int, default=-1, help="total batch size in tokens (-1 = auto-compute via Power Lines paper)")
parser.add_argument("--embedding-lr", type=float, default=0.3, help="learning rate for embedding parameters (Adam)")
parser.add_argument("--unembedding-lr", type=float, default=0.004, help="learning rate for unembedding parameters (Adam)")
parser.add_argument("--weight-decay", type=float, default=0.2, help="cautious weight decay for the Muon optimizer (for weights)")
parser.add_argument("--matrix-lr", type=float, default=0.02, help="learning rate for matrix parameters (Muon)")
parser.add_argument("--adam-beta1", type=float, default=0.8, help="Adam beta1 for embedding/unembedding")
parser.add_argument("--adam-beta2", type=float, default=0.95, help="Adam beta2 for embedding/unembedding")
parser.add_argument("--warmup-ratio", type=float, default=0.0, help="ratio of iterations for LR warmup")
parser.add_argument("--warmdown-ratio", type=float, default=0.4, help="ratio of iterations for LR warmdown")
parser.add_argument("--recur-warmup-ratio", type=float, default=0.0, help="ratio of iterations for recurrence curriculum warmup (-1.0 = auto: 1 - warmdown_ratio, 0.0 = disabled)")
parser.add_argument("--final-lr-frac", type=float, default=0.0, help="final LR as fraction of initial LR")
parser.add_argument("--resume-from-step", type=int, default=-1, help="resume training from this step (-1 = disable)")
# Evaluation
parser.add_argument("--eval-every", type=int, default=250, help="evaluate val bpb every N steps (-1 = disable)")
parser.add_argument("--eval-tokens", type=int, default=20 * 524288, help="number of tokens to evaluate val loss on")
parser.add_argument("--core-metric-every", type=int, default=2000, help="evaluate CORE metric every N steps (-1 = disable)")
parser.add_argument("--core-metric-max-per-task", type=int, default=500, help="examples per task for CORE metric")
parser.add_argument("--sample-every", type=int, default=2000, help="sample from model every N steps (-1 = disable)")
parser.add_argument("--save-every", type=int, default=-1, help="save checkpoints every N steps (-1 = only at end)")
parser.add_argument("--log-every", type=int, default=100, help="log detailed metrics to wandb every N steps")
# Output
parser.add_argument("--model-tag", type=str, default=None, help="override model tag for checkpoint directory name")
# Gradient tracking
parser.add_argument(
    "--track-gradients",
    type=str,
    choices=["none", "basic", "detailed"],
    default="basic",
    help="Gradient tracking level: none (disabled), basic (global norm), detailed (per-component norms)",
)
args = parser.parse_args()
# Resolve sentinel: recurrence warmup fills the pre-warmdown portion of training by default
if args.recur_warmup_ratio < 0:
    args.recur_warmup_ratio = 1.0 - args.warmdown_ratio
user_config = vars(args).copy()  # for logging
# -----------------------------------------------------------------------------

# Compute init
device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.
autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()
synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None
get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0
if device_type == "cuda":
    gpu_device_name = torch.cuda.get_device_name(0)
    gpu_peak_flops = get_peak_flops(gpu_device_name)
    print0(f"GPU: {gpu_device_name} | Peak FLOPS (BF16): {gpu_peak_flops:.2e}")
else:
    gpu_peak_flops = float("inf")  # MFU not meaningful for CPU/MPS

# wandb logging init
use_dummy_wandb = args.run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanochat", name=args.run, config=user_config)

# Flash Attention status
if HAS_FA3 or HAS_FA2:
    print0("✓ Using Flash Attention, efficient and awesome.")
else:
    print0("!" * 80)
    print0("WARNING: Flash Attention not available, using PyTorch SDPA fallback")
    print0("WARNING: Training will be less efficient without FA")
    if args.window_pattern != "L":
        print0(
            f"WARNING: SDPA has no support for sliding window attention (window_pattern='{args.window_pattern}'). Your GPU utilization will be terrible."
        )
        print0("WARNING: Recommend using --window-pattern L for full context attention without alternating sliding window patterns.")
    print0("!" * 80)

# Tokenizer will be useful for evaluation, also we need the vocab size
tokenizer = get_tokenizer()
token_bytes = get_token_bytes(device=device)
vocab_size = tokenizer.get_vocab_size()
print0(f"Vocab size: {vocab_size:,}")

# Model kwargs are derived from the desired size of the model
# We nudge model_dim up to the nearest multiple of head_dim to ensure clean division
# (FA3 requires head_dim divisible by 8, and this guarantees head_dim == args.head_dim exactly)
# (For very small sizes, this gives a slight "unfair" advantage to models with odd sizes)
size = args.size
base_dim = args.size * args.aspect_ratio
model_dim = ((base_dim + args.head_dim - 1) // args.head_dim) * args.head_dim
num_heads = model_dim // args.head_dim
num_kv_heads = num_heads  # default is 1:1 GQA (Group Query Attention) ratio (i.e. GQA is disabled)
head_dim = model_dim // num_heads
print0(f"size: {size}")
print0(f"model_dim: {model_dim} (base: {base_dim}, nudge: {model_dim - base_dim:+d})")
print0(f"num_heads: {num_heads}")
print0(f"head_dim: {head_dim}")
print0(f"num_kv_heads: {num_kv_heads}")

# Optimizer / data / training length related hyperparameters
tokens_per_fwdbwd = args.device_batch_size * args.max_seq_len  # tokens per iteration for a single rank
world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size  # total tokens per iteration for all ranks

# -----------------------------------------------------------------------------
# Initialize the Model

# Create a new model with random weights
model_config_kwargs = {
    "sequence_len": args.max_seq_len,
    "vocab_size": vocab_size,
    "size": size,
    "n_head": num_heads,
    "n_kv_head": num_kv_heads,
    "n_embd": model_dim,
    "window_pattern": args.window_pattern,
    # Looped Transformer config
    "n_prelude": args.n_prelude,
    "n_recur_block": args.n_recur_block,
    "n_coda": args.n_coda,
    "train_recur_mean": args.train_recur_mean,
    "train_recur_max": args.train_recur_max,
    "bptt_k": args.bptt_k,
    "input_injection": args.input_injection,
}
with torch.device("meta"):
    # All tensors are created as meta tensors (they have shape/dtype but no data)
    model_config = GPTConfig(**model_config_kwargs)
    model = GPT(model_config)
model.to_empty(device=device)  # All tensors get storage on target device but with uninitialized (garbage) data
model.init_weights()  # All tensors get initialized

# If we are resuming, overwrite the model parameters with those of the checkpoint
base_dir = get_base_dir()
output_dirname = args.model_tag if args.model_tag else f"s{args.size}"  # e.g. s12
checkpoint_dir = os.path.join(base_dir, "base_checkpoints", output_dirname)
resuming = args.resume_from_step != -1
if resuming:
    print0(f"Resuming optimization from step {args.resume_from_step}")
    model_data, optimizer_data, meta_data = load_checkpoint(
        checkpoint_dir,
        args.resume_from_step,
        device,
        load_optimizer=True,
        rank=ddp_rank,
    )
    model.load_state_dict(model_data, strict=True, assign=True)
    del model_data  # free up this memory after the copy

orig_model = (
    model  # original, uncompiled model, for saving raw model state_dict and for inference/evaluation (because the shapes may change shape)
)
if args.recur_samples_per_step:
    # Variable num_recur per step: compile blocks individually to avoid recompilation
    # from the Python loop count changing. forward() runs eagerly, compiled blocks reused.
    model.compile_blocks()
else:
    # Fixed num_recur: compile entire model as a single graph (optimal fusion)
    model = torch.compile(model)

# Detailed parameter counts
param_counts = orig_model.num_scaling_params()
print0("Parameter counts:")
for key, value in param_counts.items():
    print0(f"  {key:24s}: {value:,}")
num_params = param_counts['total']

# Effective parameters accounting for recurrent block reuse
num_effective_params = orig_model.effective_params(num_recur=int(model_config.train_recur_mean))
print0(f"Effective params: {num_effective_params:,}")

# For scaling law analysis: total params vs effective params
num_scaling_params = num_effective_params
num_flops_per_token = model.estimate_flops()
print0(f"Estimated FLOPs per token: {num_flops_per_token:e}")

# -----------------------------------------------------------------------------
# Training horizon, batch size, learning rate, and weight decay scaling
# Positive results on standard nanochat training runs; introduced by Karpathy, not specifically optimized for looped LM
# However, hyperparameter analysis showed that looped LM (r=4) is robust to standard LM params
# Refs: Power Lines (https://arxiv.org/abs/2505.13738), T_epoch (https://arxiv.org/abs/2405.13698)

# Reference s12 model: hyperparameters are tuned here and transferred to other sizes (muP style)
B_REF = 2**19  # optimal batch size at s12 ~= 524,288 tokens (measured empirically)
ref_dim = 12 * args.aspect_ratio
ref_dim = ((ref_dim + args.head_dim - 1) // args.head_dim) * args.head_dim
ref_model_kwargs = {**model_config_kwargs, "size": 12, "n_embd": ref_dim, "n_head": ref_dim // args.head_dim, "n_kv_head": ref_dim // args.head_dim}
with torch.device("meta"):
    ref_model = GPT(GPTConfig(**ref_model_kwargs))
ref_scaling_params = ref_model.effective_params(num_recur=int(model_config.train_recur_mean))
del ref_model

# 1) Calculate target training tokens from the chosen horizon method
assert args.num_iterations > 0 or args.target_param_data_ratio > 0 or args.target_flops > 0
if args.num_iterations > 0:
    assert args.total_batch_size > 0, "Must specify --total-batch-size when using --num-iterations"
    total_batch_size = args.total_batch_size
    target_tokens = args.num_iterations * total_batch_size
    num_iterations = args.num_iterations
    print0(f"Using user-provided number of iterations: {num_iterations:,}")
elif args.target_flops > 0:
    if args.recur_warmup_ratio > 0:
        # With curriculum, FLOPs/step varies. Use analytical average for initial token estimate
        # (only used to auto-compute batch size; num_iterations is set precisely below).
        avg_recur = model_config.train_recur_mean * (1.0 - 2.0 * args.recur_warmup_ratio / 3.0)
        avg_flops_per_token = model.estimate_flops(num_recur=max(1, round(avg_recur)))
        target_tokens = round(args.target_flops / avg_flops_per_token)
        print0(f"Target tokens from target FLOPs (approx, for batch size): {target_tokens:,}")
    else:
        target_tokens = round(args.target_flops / num_flops_per_token)
        print0(f"Target tokens from target FLOPs: {target_tokens:,}")
else:
    target_tokens = round(args.target_param_data_ratio * num_scaling_params)
    print0(f"Target tokens from data:param ratio {args.target_param_data_ratio}: {target_tokens:,}")

# D_REF: what the s12 reference model's training horizon would be at the same effective data:param ratio
D_REF = (target_tokens / num_scaling_params) * ref_scaling_params
print0(f"Reference training horizon (D_REF): {D_REF:,.0f} tokens")

# 2) Auto-compute optimal batch size: Bopt ∝ D^0.383 (Power Lines paper)
if args.num_iterations <= 0:
    total_batch_size = args.total_batch_size
    if total_batch_size == -1:
        batch_size_ratio = target_tokens / D_REF
        predicted_batch_size = B_REF * batch_size_ratio ** 0.383
        total_batch_size = 2 ** round(math.log2(predicted_batch_size))
        print0(f"Auto-computed optimal batch size: {total_batch_size:,} tokens")
    if args.target_flops > 0 and args.recur_warmup_ratio > 0:
        # Precise: binary search for num_iterations accounting for curriculum FLOPs
        num_iterations = solve_iterations_for_target_flops(
            orig_model.estimate_flops, args.target_flops,
            model_config.train_recur_mean, args.recur_warmup_ratio, total_batch_size,
        )
        target_tokens = num_iterations * total_batch_size
        print0(f"Calculated number of iterations from target FLOPs (with curriculum): {num_iterations:,}")
    else:
        num_iterations = round(target_tokens / total_batch_size)
        if args.target_flops > 0:
            print0(f"Calculated number of iterations from target FLOPs: {num_iterations:,}")
        else:
            print0(f"Calculated number of iterations from target data:param ratio: {num_iterations:,}")

# Gradient accumulation
assert total_batch_size % world_tokens_per_fwdbwd == 0, (
    f"total_batch_size ({total_batch_size:,}) must be divisible by tokens per micro-batch ({world_tokens_per_fwdbwd:,})"
)
grad_accum_steps = total_batch_size // world_tokens_per_fwdbwd
print0(f"Tokens / micro-batch / rank: {args.device_batch_size} x {args.max_seq_len} = {tokens_per_fwdbwd:,}")
print0(f"Tokens / micro-batch (all ranks): {world_tokens_per_fwdbwd:,}")
print0(f"Total batch size {total_batch_size:,} => gradient accumulation steps: {grad_accum_steps}")

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

# 3) Learning rate scaling: η ∝ √(B/B_ref) (sqrt scaling for AdamW, assumed for Muon too)
batch_lr_scale = 1.0
batch_ratio = total_batch_size / B_REF
if batch_ratio != 1.0:
    batch_lr_scale = batch_ratio ** 0.5
    print0(f"Scaling LRs by {batch_lr_scale:.4f} for batch size {total_batch_size:,} (reference: {B_REF:,})")

# 4) Weight decay scaling via T_epoch framework: λ = λ_ref · √(B/B_ref) · (D_ref/D)
# Central idea: T_epoch = B/(η·λ·D) should remain constant across scales.
# With η ∝ √(B/B_ref), keeping T_epoch constant requires: λ ∝ √(B/B_ref) · (D_ref/D)
# Note: Theory is for AdamW, applied to Muon as well (assumption!)
weight_decay_scaled = args.weight_decay * math.sqrt(total_batch_size / B_REF) * (D_REF / target_tokens)
if weight_decay_scaled != args.weight_decay:
    print0(f"Scaling weight decay from {args.weight_decay:.6f} to {weight_decay_scaled:.6f}")

total_tokens = total_batch_size * num_iterations
print0(f"Total number of training tokens: {total_tokens:,}")
print0(f"Tokens : Params ratio: {total_tokens / num_scaling_params:.2f}")  # Chinchilla is ~20
print0(f"Total training FLOPs estimate: {num_flops_per_token * total_tokens:e}")

# -----------------------------------------------------------------------------
# Initialize the Optimizer (combined MuonAdamW: Muon for matrix params, AdamW for rest)
adam_betas = (args.adam_beta1, args.adam_beta2)
optimizer = model.setup_optimizer(
    unembedding_lr=args.unembedding_lr * batch_lr_scale,
    embedding_lr=args.embedding_lr * batch_lr_scale,
    matrix_lr=args.matrix_lr * batch_lr_scale,
    weight_decay=weight_decay_scaled,
    adam_betas=adam_betas,
)

if resuming:
    optimizer.load_state_dict(optimizer_data)
    del optimizer_data  # free up the memory

# -----------------------------------------------------------------------------
# Initialize the DataLoaders for train/val
dataloader_resume_state_dict = None if not resuming else meta_data["dataloader_state_dict"]
train_loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(
    tokenizer,
    args.device_batch_size,
    args.max_seq_len,
    split="train",
    device=device,
    resume_state_dict=dataloader_resume_state_dict,
)


def build_val_loader():
    return tokenizing_distributed_data_loader_bos_bestfit(tokenizer, args.device_batch_size, args.max_seq_len, split="val", device=device)


x, y, dataloader_state_dict = next(train_loader)  # kick off load of the very first batch of data

# -----------------------------------------------------------------------------
# Set up hyperparameter schedulers


# Learning rate scheduler
def get_lr_multiplier(it):
    warmup_iters = round(args.warmup_ratio * num_iterations)
    warmdown_iters = round(args.warmdown_ratio * num_iterations)
    if it < warmup_iters:
        return (it + 1) / warmup_iters
    elif it <= num_iterations - warmdown_iters:
        return 1.0
    else:
        progress = (num_iterations - it) / warmdown_iters
        return progress * 1.0 + (1 - progress) * args.final_lr_frac


# Momentum scheduler for Muon optimizer
def get_muon_momentum(it):
    frac = min(it / 300, 1)
    momentum = (1 - frac) * 0.85 + frac * 0.95
    return momentum


# Weight decay scheduler for Muon optimizer (linear to zero over the course of training)
def get_weight_decay(it):
    return weight_decay_scaled * (1 - it / num_iterations)


# -----------------------------------------------------------------------------
# Loop state (variables updated by the training loop)

if not resuming:
    step = 0
    val_bpb = None  # will be set if eval_every > 0
    min_val_bpb = float("inf")
    smooth_train_loss = 0  # EMA of training loss
    total_training_time = 0  # total wall-clock time of training
    cumulative_flops = 0.0
else:
    step = meta_data["step"]
    loop_state = meta_data["loop_state"]
    val_bpb = meta_data["val_bpb"]
    min_val_bpb = loop_state["min_val_bpb"]
    smooth_train_loss = loop_state["smooth_train_loss"]
    total_training_time = loop_state["total_training_time"]
    # Recompute cumulative FLOPs from schedule (deterministic, no need to checkpoint)
    cumulative_flops = compute_cumulative_flops(
        orig_model.estimate_flops, step, num_iterations,
        model_config.train_recur_mean, args.recur_warmup_ratio, total_batch_size,
    )
    print0(f"Recomputed cumulative FLOPs up to step {step}: {cumulative_flops:e}")

# -----------------------------------------------------------------------------
# Training loop
while True:
    last_step = step == num_iterations  # loop runs num_iterations+1 times so that we can eval/save at the end
    # Curriculum: compute scheduled recurrence mean for this step
    current_recur_mean = get_scheduled_recur_mean(step, num_iterations, model_config.train_recur_mean, args.recur_warmup_ratio)
    current_flops_per_token = orig_model.estimate_flops(num_recur=int(current_recur_mean))
    flops_so_far = cumulative_flops

    # once in a while: evaluate the val bpb (all ranks participate)
    if args.eval_every > 0 and (last_step or (step > 0 and step % args.eval_every == 0)):
        model.eval()
        val_loader = build_val_loader()
        eval_steps = args.eval_tokens // (args.device_batch_size * args.max_seq_len * ddp_world_size)
        with autocast_ctx:
            val_bpb = evaluate_bpb(model, val_loader, eval_steps, token_bytes)
        print0(f"Step {step:05d} | Validation bpb: {val_bpb:.6f}")
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

    # once in a while: estimate the CORE metric (all ranks participate)
    # use the original uncompiled model because the inputs keep changing shape
    results = {}
    if args.core_metric_every > 0 and (last_step or (step > 0 and step % args.core_metric_every == 0)):
        model.eval()
        with autocast_ctx:
            results = evaluate_core(
                orig_model,
                tokenizer,
                device,
                max_per_task=args.core_metric_max_per_task,
            )
        print0(f"Step {step:05d} | CORE metric: {results['core_metric']:.4f}")
        wandb_run.log(
            {
                "step": step,
                "total_training_flops": flops_so_far,
                "core_metric": results["core_metric"],
                "centered_results": results["centered_results"],
            }
        )
        model.train()

    # once in a while: sample from the model (only on master process)
    # use the original uncompiled model because the inputs keep changing shape
    if args.sample_every > 0 and master_process and (last_step or (step > 0 and step % args.sample_every == 0)):
        model.eval()
        prompts = [
            "The capital of France is",
            "The chemical symbol of gold is",
            "If yesterday was Friday, then tomorrow will be",
            "The opposite of hot is",
            "The planets of the solar system are:",
            "My favorite color is",
            "If 5*x + 3 = 13, then x is",
        ]
        engine = Engine(orig_model, tokenizer)  # use orig_model to avoid recompilation
        for prompt in prompts:
            tokens = tokenizer(prompt, prepend="<|bos|>")
            with autocast_ctx:
                sample, _ = engine.generate_batch(tokens, num_samples=1, max_tokens=16, temperature=0)
            print0(tokenizer.decode(sample[0]))
        model.train()

    # save checkpoint: at the end of the run, or every save_every steps, except at the first step or the resume step
    if last_step or (step > 0 and step != args.resume_from_step and args.save_every > 0 and step % args.save_every == 0):
        save_checkpoint(
            checkpoint_dir,
            step,
            orig_model.state_dict(),  # model parameters
            optimizer.state_dict(),  # optimizer state
            {  # metadata saved as json
                "step": step,
                "val_bpb": val_bpb,  # loss at last step
                "model_config": model_config_kwargs,
                "user_config": user_config,  # inputs to the training script
                "device_batch_size": args.device_batch_size,
                "max_seq_len": args.max_seq_len,
                "dataloader_state_dict": dataloader_state_dict,
                "loop_state": {  # all loop state (other than step) so that we can resume training
                    "min_val_bpb": min_val_bpb,
                    "smooth_train_loss": smooth_train_loss,
                    "total_training_time": total_training_time,
                },
            },
            rank=ddp_rank,
        )

    # termination conditions (TODO: possibly also add loss explosions etc.)
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
        mean_recur=current_recur_mean,
        sigma=0.5,
        max_recur=model_config.train_recur_max,
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
        # When sampling is disabled, use the scheduled recurrence mean directly
        if num_recur is None:
            num_recur = int(current_recur_mean)
        with autocast_ctx:
            loss = model(x, y, num_recur=num_recur)
        train_loss = loss.detach()  # for logging
        loss = loss / grad_accum_steps  # each .backward() is a grad sum => normalize loss here
        loss.backward()
        x, y, dataloader_state_dict = next(train_loader)  # prefetch the next batch while the GPU is busy with forward/backward

    # Compute model health statistics: gradients and parameters (after all backward passes complete)
    model_health_stats = compute_gradient_stats(orig_model, args.track_gradients)

    # step the optimizer
    lrm = get_lr_multiplier(step)
    muon_momentum = get_muon_momentum(step)
    muon_weight_decay = get_weight_decay(step)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm
        if group["kind"] == "muon":
            group["momentum"] = muon_momentum
            group["weight_decay"] = muon_weight_decay
    optimizer.step()
    model.zero_grad(set_to_none=True)

    # Extract effective learning rates from optimizer groups (for logging)
    # Groups order: [lm_head (unembed), embedding, norms, muon_shape1, muon_shape2, ...]
    effective_lr_unembed = optimizer.param_groups[0]["lr"]  # lm_head (AdamW)
    effective_lr_embed = optimizer.param_groups[1]["lr"]    # embedding (AdamW)
    effective_lr_muon = optimizer.param_groups[3]["lr"]     # first Muon group (all Muon groups have same lr)
    train_loss_f = train_loss.item()  # .item() is a CPU-GPU sync point
    synchronize()
    t1 = time.time()
    dt = t1 - t0
    # -------------------------------------------------------------------------

    # logging (CPU action only)
    ema_beta = 0.9  # EMA decay factor for some smoothing just for nicer logging
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f  # EMA the training loss
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta ** (step + 1))  # debias the EMA
    pct_done = 100 * step / num_iterations
    tok_per_sec = int(total_batch_size / dt)
    flops_per_sec = current_flops_per_token * total_batch_size / dt
    mfu = 100 * flops_per_sec / (gpu_peak_flops * ddp_world_size)
    if step > 10:
        total_training_time += dt  # only count the time after the first 10 steps
    # Calculate ETA based on average time per step (excluding first 10 steps)
    steps_done = step - 10
    if steps_done > 0:
        avg_time_per_step = total_training_time / steps_done
        remaining_steps = num_iterations - step
        eta_seconds = remaining_steps * avg_time_per_step
        eta_str = f" | eta: {eta_seconds / 60:.1f}m"
    else:
        eta_str = ""
    epoch = dataloader_state_dict["epoch"]
    print0(
        f"step {step:05d}/{num_iterations:05d} ({pct_done:.2f}%) | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | dt: {dt * 1000:.2f}ms | tok/sec: {tok_per_sec:,} | mfu: {mfu:.2f} | epoch: {epoch} | total time: {total_training_time / 60:.2f}m{eta_str}"
    )
    if step % 100 == 0:
        # For logging: report num_recur info
        if sampled_num_recurs is None:
            logged_num_recur = int(current_recur_mean)
        else:
            logged_num_recur = sum(sampled_num_recurs) / len(sampled_num_recurs)
        log_data = {
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "train/loss": debiased_smooth_loss,
            "train/loss_raw": train_loss_f,  # raw unsmoothed loss from last microbatch
            "train/lrm": lrm,
            "train/dt": dt,
            "train/tok_per_sec": tok_per_sec,
            "train/mfu": mfu,
            "train/epoch": epoch,
            "train/num_recur": logged_num_recur,
            "train/recur_mean_scheduled": current_recur_mean,
            "lr/muon": effective_lr_muon,           # effective learning rate for Muon (matrix params)
            "lr/embed": effective_lr_embed,         # effective learning rate for embeddings
            "lr/unembed": effective_lr_unembed,     # effective learning rate for unembedding (lm_head)
            "muon/weight_decay": muon_weight_decay, # Muon weight decay schedule value
            **{f"model_health/{k}": v for k, v in model_health_stats.items()},  # Add model health stats
        }
        wandb_run.log(log_data)

    # state update
    cumulative_flops += current_flops_per_token * total_batch_size
    first_step_of_run = (step == 0) or (resuming and step == args.resume_from_step)
    step += 1

    # The garbage collector is sadly a little bit overactive and for some poorly understood reason,
    # it spends ~500ms scanning for cycles quite frequently, just to end up cleaning up very few tiny objects each time.
    # So we manually manage and help it out here
    if first_step_of_run:
        gc.collect()  # manually collect a lot of garbage from setup
        gc.freeze()  # immediately freeze all currently surviving objects and exclude them from GC
        gc.disable()  # nuclear intervention here: disable GC entirely except:
    elif step % 5000 == 0:  # every 5000 steps...
        gc.collect()  # manually collect, just to be safe for very, very long runs

# print a few more stats
print0(f"Peak memory usage: {get_max_memory() / 1024 / 1024:.2f}MiB")
print0(f"Total training time: {total_training_time / 60:.2f}m")
if val_bpb is not None:
    print0(f"Minimum validation bpb: {min_val_bpb:.6f}")

section_name = "Base model training"
if args.model_tag is not None:
    section_name += f" {args.model_tag}"
get_report().log(
    section=section_name,
    data=[
        user_config,  # CLI args
        {  # stats about the training setup
            "Number of parameters": num_params,
            "Effective parameters (w/ recur reuse)": num_effective_params,
            "Number of FLOPs per token": f"{num_flops_per_token:e}",
            "Calculated number of iterations": num_iterations,
            "Number of training tokens": total_tokens,
            "Tokens : Params ratio": total_tokens / num_params,
            "Tokens : Effective Params ratio": total_tokens / num_effective_params,
            "DDP world size": ddp_world_size,
            "warmup_ratio": args.warmup_ratio,
            "warmdown_ratio": args.warmdown_ratio,
            "final_lr_frac": args.final_lr_frac,
        },
        {  # stats about training outcomes
            "Minimum validation bpb": min_val_bpb if val_bpb is not None else None,
            "Final validation bpb": val_bpb,
            "CORE metric estimate": results.get("core_metric", None),
            "MFU %": f"{mfu:.2f}%",
            "Total training flops": f"{flops_so_far:e}",
            "Total training time": f"{total_training_time / 60:.2f}m",
            "Peak memory usage": f"{get_max_memory() / 1024 / 1024:.2f}MiB",
        },
    ],
)

# cleanup
wandb_run.finish()  # wandb run finish
compute_cleanup()
