"""
Early-exit gating analysis for looped transformers.

Evaluates training-free early-exit gates using model.forward_gated, which
performs actual gated forward passes with state freezing and early exit.
Produces efficiency vs accuracy Pareto curves comparing across models.

Two evaluation modes:
  --val-loss: BPB on validation data (SFT val mixture for -i sft, FineWeb for -i base)
  -a TASK:   Accuracy on categorical benchmarks (ARC, MMLU)

Gate functions (all zero-shot, no trained params, defined in nanochat/gates.py):
  1. Acceleration (Pappone et al.): normalized second-order convergence, two-hit
  2. Relative L2: ||s_i - s_{i-1}|| / ||s_i||
  3. KL divergence on intermediate logits

Example:
    uv run python dev/analysis/gating_analysis.py -i sft --val-loss --val-tokens 3000000 -g d12
    uv run python dev/analysis/gating_analysis.py -i sft -a ARC-Easy -g d12
    uv run python dev/analysis/gating_analysis.py -i sft -a "ARC-Easy|ARC-Challenge|MMLU" -g "d12|d20" --max-problems 200
"""

import argparse
import csv
import math
from contextlib import nullcontext
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.distributed as dist
import torch.nn.functional as F

from nanochat.checkpoint_manager import load_model
from nanochat.common import autodetect_device_type, compute_cleanup, compute_init, get_base_dir, print0
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
from nanochat.gates import GateConfig
from nanochat.tokenizer import get_token_bytes
from tasks.arc import ARC
from tasks.common import TaskMixture
from tasks.gsm8k import GSM8K
from tasks.mmlu import MMLU
from tasks.smoltalk import SmolTalk

# ---------------------------------------------------------------------------
# Task registry (categorical tasks only — gate analysis uses logit probing)

TASK_MODULES = {
    "ARC-Easy": partial(ARC, subset="ARC-Easy", split="test"),
    "ARC-Challenge": partial(ARC, subset="ARC-Challenge", split="test"),
    "MMLU": partial(MMLU, subset="all", split="test"),
}

# ---------------------------------------------------------------------------
# Gate sweep configuration: (name, thresholds)

GATES = [
    #("acceleration", [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0]),
    ("relative_l2", [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]),
    ("kl_divergence", [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 0.75]),
]

# ---------------------------------------------------------------------------
# SFT validation data generator (mirrors chat_sft.py's sft_data_generator_bos_bestfit)


def sft_val_data_generator(tokenizer, batch_size: int, sequence_len: int, device: str):
    """Yield (inputs, targets) batches from the SFT validation mixture.

    Uses the same TaskMixture and bestfit packing as chat_sft.py so that val BPB
    is measured on the distribution the SFT model was actually trained on.
    """
    val_dataset = TaskMixture([
        SmolTalk(split="test"),
        MMLU(subset="all", split="test", stop=5200),
        GSM8K(subset="main", split="test", stop=420),
    ])

    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    row_capacity = sequence_len + 1
    bos_token = tokenizer.get_bos_token_id()
    dataset_size = len(val_dataset)
    buffer_size = 100
    conv_buffer: list[list[int]] = []
    cursor = rank
    use_cuda = device == "cuda"

    def refill_buffer():
        nonlocal cursor
        while len(conv_buffer) < buffer_size:
            conversation = val_dataset[cursor]
            ids, _ = tokenizer.render_conversation(conversation)
            conv_buffer.append(ids)
            cursor += world_size
            if cursor >= dataset_size:
                cursor = cursor % dataset_size

    while True:
        rows = []
        row_lengths = []
        for _ in range(batch_size):
            row: list[int] = []
            padded = False
            while len(row) < row_capacity:
                while len(conv_buffer) < buffer_size:
                    refill_buffer()

                remaining = row_capacity - len(row)
                best_idx = -1
                best_len = 0
                for i, conv in enumerate(conv_buffer):
                    if len(conv) <= remaining and len(conv) > best_len:
                        best_idx = i
                        best_len = len(conv)

                if best_idx >= 0:
                    row.extend(conv_buffer.pop(best_idx))
                else:
                    content_len = len(row)
                    row.extend([bos_token] * remaining)
                    padded = True
                    break

            row_lengths.append(content_len if padded else row_capacity)
            rows.append(row[:row_capacity])

        batch_tensor = torch.tensor(rows, dtype=torch.long, pin_memory=use_cuda)
        inputs = batch_tensor[:, :-1].to(device=device, dtype=torch.int32, non_blocking=use_cuda)
        targets = batch_tensor[:, 1:].to(device=device, dtype=torch.int64, non_blocking=use_cuda)

        for i, cl in enumerate(row_lengths):
            if cl < row_capacity:
                targets[i, cl - 1:] = -1

        yield inputs, targets


# ---------------------------------------------------------------------------
# Evaluation using model.forward_gated


@torch.no_grad()
def evaluate_with_gates(
    model,
    tokenizer,
    task_object,
    batch_size: int = 8,
    num_recur: int | None = None,
    max_problems: int | None = None,
) -> dict:
    """
    Run categorical eval using model.forward_gated for each gate/threshold.

    Runs the actual gated forward pass (with state freezing and early exit)
    for each gate/threshold combination. One forward pass per combination per
    batch, so slower than post-hoc simulation but gives true gated accuracy.

    Returns a dict with:
        - gate_results: {gate_name: {threshold: {"correct": int, "total": int, "total_depth": int}}}
        - max_r_correct, r1_correct, train_r_correct, total, num_recur, train_recur
    """
    device = model.get_device()
    bos = tokenizer.get_bos_token_id()
    train_recur = int(model.config.train_recur_mean)

    if num_recur is None:
        num_recur = train_recur

    num_problems = len(task_object) if max_problems is None else min(len(task_object), max_problems)

    # Initialize gate result accumulators
    gate_results = {}
    for gate_name, thresholds in GATES:
        gate_results[gate_name] = {}
        for thr in thresholds:
            gate_results[gate_name][thr] = {"correct": 0, "total": 0, "total_depth": 0}

    max_r_correct = 0
    r1_correct = 0
    train_r_correct = 0
    total = 0
    letter_to_id_cache = {}

    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    num_batches = -(-num_problems // batch_size)
    total_gate_combos = sum(len(thrs) for _, thrs in GATES)

    for batch_idx in range(rank, num_batches, world_size):
        i0 = batch_idx * batch_size
        i1 = min((batch_idx + 1) * batch_size, num_problems)

        conversations = [task_object[ii] for ii in range(i0, i1)]
        prompt_ids = [tokenizer.render_for_completion(conv) for conv in conversations]
        max_length = max(len(ids) for ids in prompt_ids)
        answer_positions = [len(ids) - 1 for ids in prompt_ids]
        padded = [ids + [bos] * (max_length - len(ids)) for ids in prompt_ids]
        input_ids = torch.tensor(padded, dtype=torch.long, device=device)

        # Baselines from standard forward (one pass for max_r, r1, train_r)
        logits_full, _s, intermediate_logits = model(
            input_ids, num_recur=num_recur, return_intermediate_logits=True,
        )

        # Run forward_gated for each gate/threshold
        gated_results: dict[str, dict[float, tuple]] = {}
        for gate_name, thresholds in GATES:
            gated_results[gate_name] = {}
            for thr in thresholds:
                gate_config = GateConfig(gate_type=gate_name, threshold=thr)
                logits_g, _s_g, stats = model.forward_gated(
                    input_ids, gate_config, num_recur=num_recur,
                )
                gated_results[gate_name][thr] = (logits_g, stats)

        # Evaluate each example
        for idx, conversation in enumerate(conversations):
            letters = conversation["letters"]
            letter_ids = []
            for letter in letters:
                if letter not in letter_to_id_cache:
                    encoded = tokenizer.encode(letter)
                    assert len(encoded) == 1
                    letter_to_id_cache[letter] = encoded[0]
                letter_ids.append(letter_to_id_cache[letter])

            answer_pos = answer_positions[idx]
            correct_letter = conversation["messages"][-1]["content"]
            correct_idx_in_letters = letters.index(correct_letter)

            # Max-r baseline
            focus_logits = logits_full[idx, answer_pos, letter_ids]
            max_r_correct += int(focus_logits.argmax(dim=-1).item() == correct_idx_in_letters)

            # r=1 baseline
            r1_focus = intermediate_logits[0][idx, answer_pos, letter_ids]
            r1_correct += int(r1_focus.argmax(dim=-1).item() == correct_idx_in_letters)

            # train_r baseline
            train_r_focus = intermediate_logits[train_recur - 1][idx, answer_pos, letter_ids]
            train_r_correct += int(train_r_focus.argmax(dim=-1).item() == correct_idx_in_letters)

            # Gate evaluations from forward_gated
            for gate_name, thresholds in GATES:
                for thr in thresholds:
                    logits_g, stats = gated_results[gate_name][thr]
                    g_focus = logits_g[idx, answer_pos, letter_ids]
                    is_correct = g_focus.argmax(dim=-1).item() == correct_idx_in_letters
                    token_exit = stats.exit_depths[idx, answer_pos].item()

                    gate_results[gate_name][thr]["correct"] += int(is_correct)
                    gate_results[gate_name][thr]["total"] += 1
                    gate_results[gate_name][thr]["total_depth"] += token_exit

            total += 1

        print0(
            f"\r\033[KBatch {batch_idx + 1}/{num_batches} "
            f"({total_gate_combos} gated fwd/batch) | "
            f"max_r acc: {max_r_correct}/{total} ({100 * max_r_correct / total:.1f}%)",
            end="",
        )

    print0()

    # Reduce across ranks if distributed
    if world_size > 1:
        counters = torch.tensor(
            [max_r_correct, r1_correct, train_r_correct, total],
            dtype=torch.long, device=device,
        )
        dist.all_reduce(counters, op=dist.ReduceOp.SUM)
        max_r_correct, r1_correct, train_r_correct, total = counters.tolist()

        for gate_name, thresholds_dict in gate_results.items():
            gate_counters = []
            gate_keys = sorted(thresholds_dict.keys())
            for thr in gate_keys:
                s = thresholds_dict[thr]
                gate_counters.extend([s["correct"], s["total"], s["total_depth"]])
            gate_tensor = torch.tensor(gate_counters, dtype=torch.long, device=device)
            dist.all_reduce(gate_tensor, op=dist.ReduceOp.SUM)
            gate_list = gate_tensor.tolist()
            for i, thr in enumerate(gate_keys):
                thresholds_dict[thr]["correct"] = gate_list[i * 3]
                thresholds_dict[thr]["total"] = gate_list[i * 3 + 1]
                thresholds_dict[thr]["total_depth"] = gate_list[i * 3 + 2]

    return {
        "gate_results": gate_results,
        "max_r_correct": max_r_correct,
        "r1_correct": r1_correct,
        "train_r_correct": train_r_correct,
        "total": total,
        "num_recur": num_recur,
        "train_recur": train_recur,
    }


# ---------------------------------------------------------------------------
# Validation loss (BPB) evaluation with gates


def _bpb_from_nats_bytes(total_nats: float, total_bytes: int) -> float:
    if total_bytes == 0:
        return float("inf")
    return total_nats / (math.log(2) * total_bytes)


def _accumulate_bpb(loss2d: torch.Tensor, targets: torch.Tensor, token_bytes: torch.Tensor):
    """Accumulate nats and bytes from per-token losses, masking special tokens."""
    loss_flat = loss2d.view(-1)
    y_flat = targets.view(-1)
    valid = y_flat >= 0
    y_safe = torch.where(valid, y_flat, torch.zeros_like(y_flat))
    num_bytes = torch.where(valid, token_bytes[y_safe], torch.zeros_like(y_flat, dtype=token_bytes.dtype))
    counted = num_bytes > 0
    nats = (loss_flat * counted).sum()
    byte_count = num_bytes.sum()
    return nats, byte_count


@torch.no_grad()
def evaluate_bpb_with_gates(
    model,
    val_loader,
    token_bytes: torch.Tensor,
    batch_size: int,
    num_recur: int,
    val_tokens: int,
) -> dict:
    """
    Compute BPB on validation data for baseline recursion depths and each gate/threshold.

    Evaluates `val_tokens` total tokens (rounded down to full batches). For each batch:
      - 3 baseline forwards (max_r, train_r, r=1) via model(x, y)
      - 1 gated forward per gate/threshold via model.forward_gated

    Args:
        val_loader: Iterator yielding (inputs, targets) batches. Use the pretraining
            loader for base models and the SFT loader for SFT models.

    Returns dict with:
        - gate_results: {gate_name: {thr: {"total_nats", "total_bytes", "total_depth", "total_tokens"}}}
        - max_r_bpb, train_r_bpb, r1_bpb, num_recur, train_recur
    """
    train_recur = int(model.config.train_recur_mean)
    sequence_len = model.config.sequence_len
    tokens_per_step = batch_size * sequence_len
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    steps = val_tokens // (tokens_per_step * world_size)
    assert steps >= 1, f"val_tokens={val_tokens} too small for batch_size={batch_size} * seq_len={sequence_len} * world_size={world_size}"
    print0(f"  val_tokens={val_tokens} -> {steps} steps x {tokens_per_step} tok/step x {world_size} ranks")

    loader_iter = iter(val_loader)

    # Baseline accumulators
    baselines = {
        "max_r": {"nats": 0.0, "bytes": 0},
        "train_r": {"nats": 0.0, "bytes": 0},
        "r1": {"nats": 0.0, "bytes": 0},
    }
    # Gate accumulators
    gate_results: dict[str, dict[float, dict]] = {}
    for gate_name, thresholds in GATES:
        gate_results[gate_name] = {}
        for thr in thresholds:
            gate_results[gate_name][thr] = {
                "total_nats": 0.0, "total_bytes": 0,
                "total_depth": 0, "total_depth_sq": 0.0, "total_tokens": 0,
                "depth_hist": None,  # (num_recur,) counts, initialized on first batch
            }

    total_gate_combos = sum(len(thrs) for _, thrs in GATES)

    for step_idx in range(steps):
        x, y = next(loader_iter)
        B, T = x.shape

        # Baseline forwards (each returns per-token loss)
        loss_max_r = model(x, y, loss_reduction="none", num_recur=num_recur)
        nats, nbytes = _accumulate_bpb(loss_max_r, y, token_bytes)
        baselines["max_r"]["nats"] += nats.item()
        baselines["max_r"]["bytes"] += nbytes.item()

        if train_recur != num_recur:
            loss_train_r = model(x, y, loss_reduction="none", num_recur=train_recur)
        else:
            loss_train_r = loss_max_r
        nats, nbytes = _accumulate_bpb(loss_train_r, y, token_bytes)
        baselines["train_r"]["nats"] += nats.item()
        baselines["train_r"]["bytes"] += nbytes.item()

        loss_r1 = model(x, y, loss_reduction="none", num_recur=1)
        nats, nbytes = _accumulate_bpb(loss_r1, y, token_bytes)
        baselines["r1"]["nats"] += nats.item()
        baselines["r1"]["bytes"] += nbytes.item()

        # Gated forwards
        for gate_name, thresholds in GATES:
            for thr in thresholds:
                gate_config = GateConfig(gate_type=gate_name, threshold=thr)
                logits_g, _s_g, stats = model.forward_gated(x, gate_config, num_recur=num_recur)
                loss_g = F.cross_entropy(
                    logits_g.view(-1, logits_g.size(-1)), y.view(-1),
                    ignore_index=-1, reduction="none",
                ).view(B, T)
                nats, nbytes = _accumulate_bpb(loss_g, y, token_bytes)
                depths_flat = stats.exit_depths.view(-1).long()
                acc = gate_results[gate_name][thr]
                acc["total_nats"] += nats.item()
                acc["total_bytes"] += nbytes.item()
                acc["total_depth"] += depths_flat.sum().item()
                acc["total_depth_sq"] += (depths_flat.float() ** 2).sum().item()
                acc["total_tokens"] += B * T
                batch_hist = torch.bincount(depths_flat, minlength=num_recur + 1)[:num_recur + 1]
                if acc["depth_hist"] is None:
                    acc["depth_hist"] = batch_hist
                else:
                    acc["depth_hist"] += batch_hist

        max_r_bpb = _bpb_from_nats_bytes(baselines["max_r"]["nats"], baselines["max_r"]["bytes"])
        print0(
            f"\r\033[KStep {step_idx + 1}/{steps} "
            f"({total_gate_combos} gated fwd/step) | "
            f"max_r bpb: {max_r_bpb:.4f}",
            end="",
        )

    print0()

    # Reduce across ranks
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    if world_size > 1:
        device_t = model.get_device()
        # Baselines
        bl_tensor = torch.tensor(
            [baselines[k][m] for k in ["max_r", "train_r", "r1"] for m in ["nats", "bytes"]],
            dtype=torch.float64, device=device_t,
        )
        dist.all_reduce(bl_tensor, op=dist.ReduceOp.SUM)
        bl_list = bl_tensor.tolist()
        for i, k in enumerate(["max_r", "train_r", "r1"]):
            baselines[k]["nats"] = bl_list[i * 2]
            baselines[k]["bytes"] = int(bl_list[i * 2 + 1])

        # Gates
        for gate_name in gate_results:
            gate_keys = sorted(gate_results[gate_name].keys())
            vals = []
            for thr in gate_keys:
                s = gate_results[gate_name][thr]
                vals.extend([s["total_nats"], s["total_bytes"], s["total_depth"], s["total_depth_sq"], s["total_tokens"]])
            gt = torch.tensor(vals, dtype=torch.float64, device=device_t)
            dist.all_reduce(gt, op=dist.ReduceOp.SUM)
            gl = gt.tolist()
            for i, thr in enumerate(gate_keys):
                gate_results[gate_name][thr]["total_nats"] = gl[i * 5]
                gate_results[gate_name][thr]["total_bytes"] = int(gl[i * 5 + 1])
                gate_results[gate_name][thr]["total_depth"] = gl[i * 5 + 2]
                gate_results[gate_name][thr]["total_depth_sq"] = gl[i * 5 + 3]
                gate_results[gate_name][thr]["total_tokens"] = int(gl[i * 5 + 4])
            # Reduce histograms
            for thr in gate_keys:
                h = gate_results[gate_name][thr]["depth_hist"]
                if h is not None:
                    h = h.to(device=device_t)
                    dist.all_reduce(h, op=dist.ReduceOp.SUM)
                    gate_results[gate_name][thr]["depth_hist"] = h

    return {
        "gate_results": gate_results,
        "max_r_bpb": _bpb_from_nats_bytes(baselines["max_r"]["nats"], baselines["max_r"]["bytes"]),
        "train_r_bpb": _bpb_from_nats_bytes(baselines["train_r"]["nats"], baselines["train_r"]["bytes"]),
        "r1_bpb": _bpb_from_nats_bytes(baselines["r1"]["nats"], baselines["r1"]["bytes"]),
        "num_recur": num_recur,
        "train_recur": train_recur,
    }


# ---------------------------------------------------------------------------
# FLOPs fraction computation


def compute_flops_fraction(scaling_params: dict, exit_depth: float, train_recur: int) -> float:
    """Compute FLOPs fraction: (fixed + r_exit * recur) / (fixed + train_r * recur).

    Normalized so fraction=1.0 at train_r. Values <1 save FLOPs, >1 use more.
    """
    fixed = scaling_params["prelude"] + scaling_params["coda"] + scaling_params["wte"] + scaling_params["lm_head"] + scaling_params["scalars"]
    recur = scaling_params["recur_block"] + scaling_params["inject"]
    train_cost = fixed + train_recur * recur
    exit_cost = fixed + exit_depth * recur
    return exit_cost / train_cost


# ---------------------------------------------------------------------------
# Plotting


def plot_pareto_curves(
    all_model_results: dict[str, dict],
    task_name: str,
    output_dir: Path,
    suffix: str = "",
):
    """
    Plot FLOPs fraction vs accuracy Pareto curves.

    One plot per task. Each gate is a curve (connected scatter). Baselines
    shown as horizontal/vertical lines. Faceted or overlaid across models.
    """
    num_models = len(all_model_results)
    fig, axes = plt.subplots(1, num_models, figsize=(7 * num_models, 6), squeeze=False)

    # Colorblind-safe palette (Wong)
    gate_colors = {
        "acceleration": "#332288",
        "relative_l2": "#E69F00",
        "kl_divergence": "#009E73",
    }
    gate_markers = {
        "acceleration": "o",
        "relative_l2": "s",
        "kl_divergence": "D",
    }

    for col_idx, (model_tag, result) in enumerate(all_model_results.items()):
        ax = axes[0, col_idx]
        params = result["scaling_params"]
        num_recur = result["eval_result"]["num_recur"]
        train_recur = result["eval_result"]["train_recur"]
        total = result["eval_result"]["total"]
        max_r_acc = result["eval_result"]["max_r_correct"] / total
        r1_acc = result["eval_result"]["r1_correct"] / total
        train_r_acc = result["eval_result"]["train_r_correct"] / total

        # Baselines (FLOPs normalized to train_r=1.0)
        r1_flops = compute_flops_fraction(params, 1, train_recur)
        max_r_flops = compute_flops_fraction(params, num_recur, train_recur)
        ax.axhline(y=max_r_acc, color="black", linestyle="--", alpha=0.5, label=f"max_r={num_recur} ({max_r_acc:.3f})")
        ax.axvline(x=max_r_flops, color="black", linestyle="--", alpha=0.3)
        ax.axhline(y=train_r_acc, color="#CC79A7", linestyle="-.", alpha=0.5, label=f"train_r={train_recur} ({train_r_acc:.3f})")
        ax.axvline(x=1.0, color="#CC79A7", linestyle="-.", alpha=0.3)
        ax.axhline(y=r1_acc, color="gray", linestyle=":", alpha=0.5, label=f"r=1 ({r1_acc:.3f})")
        ax.axvline(x=r1_flops, color="gray", linestyle=":", alpha=0.3)

        # Gate curves
        for gate_name, thresholds_dict in result["eval_result"]["gate_results"].items():
            flops_list = []
            acc_list = []
            for thr, stats in sorted(thresholds_dict.items()):
                if stats["total"] == 0:
                    continue
                acc = stats["correct"] / stats["total"]
                mean_depth = stats["total_depth"] / stats["total"]
                flops_frac = compute_flops_fraction(params, mean_depth, train_recur)
                flops_list.append(flops_frac)
                acc_list.append(acc)

            if flops_list:
                ax.plot(
                    flops_list, acc_list,
                    color=gate_colors[gate_name],
                    marker=gate_markers[gate_name],
                    markersize=5,
                    label=gate_name,
                    linewidth=1.5,
                    alpha=0.8,
                )

        ax.set_xlabel("FLOPs Fraction", fontsize=11)
        ax.set_ylabel("Accuracy", fontsize=11)
        ax.set_title(f"{task_name} — {model_tag} (r={num_recur})", fontsize=12)
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / f"gating_pareto_{task_name.lower().replace('-', '_')}{suffix}.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print0(f"Plot saved to: {output_path}")
    plt.close()


# ---------------------------------------------------------------------------
# CSV output (cumulative — appends to existing file, skips duplicate keys)


def _load_existing_csv_keys(csv_path: Path, key_columns: list[int]) -> set[tuple]:
    """Load existing CSV and return set of key tuples for deduplication."""
    existing: set[tuple] = set()
    if not csv_path.exists():
        return existing
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header
        for row in reader:
            if row:
                existing.add(tuple(row[i] for i in key_columns))
    return existing


def save_results_csv(
    all_model_results: dict[str, dict],
    task_name: str,
    output_dir: Path,
    suffix: str = "",
):
    """Save raw results as CSV for downstream analysis (cumulative append)."""
    csv_path = output_dir / f"gating_results_{task_name.lower().replace('-', '_')}{suffix}.csv"
    header = [
        "model_tag", "task", "gate", "threshold",
        "accuracy", "mean_exit_depth", "flops_fraction",
        "correct", "total",
    ]
    # Key: (model_tag, task, gate, threshold)
    existing = _load_existing_csv_keys(csv_path, [0, 1, 2, 3])
    write_header = not csv_path.exists()
    new_rows = 0

    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)

        for model_tag, result in all_model_results.items():
            params = result["scaling_params"]
            num_recur = result["eval_result"]["num_recur"]
            total = result["eval_result"]["total"]
            train_recur = result["eval_result"]["train_recur"]

            rows = []
            max_r_acc = result["eval_result"]["max_r_correct"] / total
            max_r_flops = compute_flops_fraction(params, num_recur, train_recur)
            rows.append([model_tag, task_name, "max_r", num_recur, f"{max_r_acc:.6f}", num_recur, f"{max_r_flops:.6f}", result["eval_result"]["max_r_correct"], total])

            train_r_acc = result["eval_result"]["train_r_correct"] / total
            rows.append([model_tag, task_name, "train_r", train_recur, f"{train_r_acc:.6f}", train_recur, "1.000000", result["eval_result"]["train_r_correct"], total])

            r1_acc = result["eval_result"]["r1_correct"] / total
            r1_flops = compute_flops_fraction(params, 1, train_recur)
            rows.append([model_tag, task_name, "r1", 1, f"{r1_acc:.6f}", 1, f"{r1_flops:.6f}", result["eval_result"]["r1_correct"], total])

            for gate_name, thresholds_dict in result["eval_result"]["gate_results"].items():
                for thr, stats in sorted(thresholds_dict.items()):
                    if stats["total"] == 0:
                        continue
                    acc = stats["correct"] / stats["total"]
                    mean_depth = stats["total_depth"] / stats["total"]
                    flops_frac = compute_flops_fraction(params, mean_depth, train_recur)
                    rows.append([model_tag, task_name, gate_name, thr, f"{acc:.6f}", f"{mean_depth:.2f}", f"{flops_frac:.6f}", stats["correct"], stats["total"]])

            for row in rows:
                key = tuple(str(row[i]) for i in [0, 1, 2, 3])
                if key not in existing:
                    writer.writerow(row)
                    existing.add(key)
                    new_rows += 1

    print0(f"CSV saved to: {csv_path} ({new_rows} new rows)")


# ---------------------------------------------------------------------------
# BPB plotting and CSV


def plot_pareto_bpb(
    all_model_results: dict[str, dict],
    output_dir: Path,
    suffix: str = "",
):
    """
    Plot FLOPs fraction vs BPB Pareto curves (val loss).

    One subplot per model. Lower BPB is better.
    """
    num_models = len(all_model_results)
    fig, axes = plt.subplots(1, num_models, figsize=(7 * num_models, 6), squeeze=False)

    gate_colors = {
        "acceleration": "#332288",
        "relative_l2": "#E69F00",
        "kl_divergence": "#009E73",
    }
    gate_markers = {
        "acceleration": "o",
        "relative_l2": "s",
        "kl_divergence": "D",
    }

    for col_idx, (model_tag, result) in enumerate(all_model_results.items()):
        ax = axes[0, col_idx]
        params = result["scaling_params"]
        bpb_result = result["bpb_result"]
        num_recur = bpb_result["num_recur"]
        train_recur = bpb_result["train_recur"]

        max_r_bpb = bpb_result["max_r_bpb"]
        train_r_bpb = bpb_result["train_r_bpb"]
        r1_bpb = bpb_result["r1_bpb"]

        # Baselines (FLOPs normalized to train_r=1.0)
        r1_flops = compute_flops_fraction(params, 1, train_recur)
        max_r_flops = compute_flops_fraction(params, num_recur, train_recur)
        ax.axhline(y=max_r_bpb, color="black", linestyle="--", alpha=0.5, label=f"max_r={num_recur} ({max_r_bpb:.4f})")
        ax.axvline(x=max_r_flops, color="black", linestyle="--", alpha=0.3)
        ax.axhline(y=train_r_bpb, color="#CC79A7", linestyle="-.", alpha=0.5, label=f"train_r={train_recur} ({train_r_bpb:.4f})")
        ax.axvline(x=1.0, color="#CC79A7", linestyle="-.", alpha=0.3)
        ax.axhline(y=r1_bpb, color="gray", linestyle=":", alpha=0.5, label=f"r=1 ({r1_bpb:.4f})")
        ax.axvline(x=r1_flops, color="gray", linestyle=":", alpha=0.3)

        # Gate curves
        for gate_name, thresholds_dict in bpb_result["gate_results"].items():
            flops_list = []
            bpb_list = []
            for thr, stats in sorted(thresholds_dict.items()):
                if stats["total_tokens"] == 0:
                    continue
                bpb = _bpb_from_nats_bytes(stats["total_nats"], stats["total_bytes"])
                mean_depth = stats["total_depth"] / stats["total_tokens"]
                flops_frac = compute_flops_fraction(params, mean_depth, train_recur)
                flops_list.append(flops_frac)
                bpb_list.append(bpb)

            if flops_list:
                ax.plot(
                    flops_list, bpb_list,
                    color=gate_colors[gate_name],
                    marker=gate_markers[gate_name],
                    markersize=5,
                    label=gate_name,
                    linewidth=1.5,
                    alpha=0.8,
                )

        ax.set_xlabel("FLOPs Fraction", fontsize=11)
        ax.set_ylabel("BPB (val)", fontsize=11)
        ax.set_title(f"Val Loss — {model_tag} (r={num_recur})", fontsize=12)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / f"gating_pareto_val_bpb{suffix}.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print0(f"Plot saved to: {output_path}")
    plt.close()


def plot_depth_histograms(
    all_model_results: dict[str, dict],
    output_dir: Path,
    suffix: str = "",
):
    """Plot exit depth distributions: one subplot per (gate, model) combo."""
    import numpy as np

    gate_names = [g for g, _ in GATES]
    model_tags = list(all_model_results.keys())
    num_gates = len(gate_names)
    num_models = len(model_tags)

    fig, axes = plt.subplots(
        num_gates, num_models,
        figsize=(6 * num_models, 4 * num_gates),
        squeeze=False,
    )

    # Colorblind-safe sequential palette for thresholds (light to dark)
    cmap = plt.cm.viridis

    for row, gate_name in enumerate(gate_names):
        for col, model_tag in enumerate(model_tags):
            ax = axes[row, col]
            bpb_result = all_model_results[model_tag]["bpb_result"]
            num_recur = bpb_result["num_recur"]

            if gate_name not in bpb_result["gate_results"]:
                ax.set_visible(False)
                continue

            thresholds_dict = bpb_result["gate_results"][gate_name]
            sorted_thrs = sorted(thresholds_dict.keys())
            n_thrs = len(sorted_thrs)
            depths = np.arange(1, num_recur + 1)
            bar_width = 0.8 / n_thrs

            for i, thr in enumerate(sorted_thrs):
                hist = thresholds_dict[thr].get("depth_hist")
                if hist is None:
                    continue
                hist_np = hist.cpu().numpy().astype(float)
                # depth_hist is indexed 0..num_recur; depths are 1..num_recur
                counts = hist_np[1:num_recur + 1]
                total = counts.sum()
                if total == 0:
                    continue
                fracs = counts / total
                color = cmap(i / max(n_thrs - 1, 1))
                ax.bar(
                    depths + (i - n_thrs / 2) * bar_width,
                    fracs, width=bar_width,
                    color=color, alpha=0.85,
                    label=f"thr={thr}",
                )

            ax.set_xlabel("Exit Depth", fontsize=10)
            ax.set_ylabel("Fraction of Tokens", fontsize=10)
            ax.set_title(f"{gate_name} — {model_tag}", fontsize=11)
            ax.set_xticks(depths)
            ax.legend(fontsize=7, loc="upper right")
            ax.grid(True, alpha=0.2, axis="y")

    plt.tight_layout()
    output_path = output_dir / f"gating_depth_hist{suffix}.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print0(f"Plot saved to: {output_path}")
    plt.close()


def save_bpb_csv(
    all_model_results: dict[str, dict],
    output_dir: Path,
    suffix: str = "",
):
    """Save BPB gating results as CSV (cumulative append)."""
    csv_path = output_dir / f"gating_results_val_bpb{suffix}.csv"
    header = [
        "model_tag", "gate", "threshold",
        "bpb", "mean_exit_depth", "std_exit_depth", "flops_fraction",
    ]
    # Key: (model_tag, gate, threshold)
    existing = _load_existing_csv_keys(csv_path, [0, 1, 2])
    write_header = not csv_path.exists()
    new_rows = 0

    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)

        for model_tag, result in all_model_results.items():
            params = result["scaling_params"]
            bpb_result = result["bpb_result"]
            num_recur = bpb_result["num_recur"]
            train_recur = bpb_result["train_recur"]

            rows = []
            # Baselines (std=0 for fixed-depth baselines)
            max_r_flops = compute_flops_fraction(params, num_recur, train_recur)
            rows.append([model_tag, "max_r", num_recur, f"{bpb_result['max_r_bpb']:.6f}", num_recur, "0.00", f"{max_r_flops:.6f}"])
            rows.append([model_tag, "train_r", train_recur, f"{bpb_result['train_r_bpb']:.6f}", train_recur, "0.00", "1.000000"])
            r1_flops = compute_flops_fraction(params, 1, train_recur)
            rows.append([model_tag, "r1", 1, f"{bpb_result['r1_bpb']:.6f}", 1, "0.00", f"{r1_flops:.6f}"])

            # Gate results
            for gate_name, thresholds_dict in bpb_result["gate_results"].items():
                for thr, stats in sorted(thresholds_dict.items()):
                    if stats["total_tokens"] == 0:
                        continue
                    bpb = _bpb_from_nats_bytes(stats["total_nats"], stats["total_bytes"])
                    mean_depth = stats["total_depth"] / stats["total_tokens"]
                    var_depth = stats["total_depth_sq"] / stats["total_tokens"] - mean_depth ** 2
                    std_depth = max(var_depth, 0.0) ** 0.5
                    flops_frac = compute_flops_fraction(params, mean_depth, train_recur)
                    rows.append([model_tag, gate_name, thr, f"{bpb:.6f}", f"{mean_depth:.2f}", f"{std_depth:.2f}", f"{flops_frac:.6f}"])

            for row in rows:
                key = tuple(str(row[i]) for i in [0, 1, 2])
                if key not in existing:
                    writer.writerow(row)
                    existing.add(key)
                    new_rows += 1

    print0(f"CSV saved to: {csv_path} ({new_rows} new rows)")


# ---------------------------------------------------------------------------
# Main


def main():
    parser = argparse.ArgumentParser(description="Early-exit gating analysis for looped transformers")
    parser.add_argument("-i", "--source", type=str, required=True, help="Model source: base|sft|rl")
    parser.add_argument("-a", "--task-name", type=str, default=None, help="Pipe-separated task names (e.g., ARC-Easy|MMLU)")
    parser.add_argument("--val-loss", action="store_true", help="Evaluate gated BPB on validation data")
    parser.add_argument("--val-tokens", type=int, default=3_000_000, help="Total val tokens for --val-loss (default: 3M)")
    parser.add_argument("-g", "--model-tags", type=str, default=None, help="Pipe-separated model tags (e.g., d12|d20). Default: largest.")
    parser.add_argument("-r", "--num-recur", type=int, default=None, help="Max recurrences (default: model default)")
    parser.add_argument("-b", "--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("-x", "--max-problems", type=int, default=None, help="Max problems to evaluate (categorical only)")
    parser.add_argument("-d", "--dtype", type=str, default="bfloat16", choices=["float32", "bfloat16"])
    parser.add_argument("--device-type", type=str, default="", choices=["cuda", "cpu", "mps", ""])
    args = parser.parse_args()

    if not args.val_loss and args.task_name is None:
        parser.error("Specify --val-loss and/or -a TASK_NAME")

    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    _ddp, _ddp_rank, _ddp_local_rank, _ddp_world_size, device = compute_init(device_type)
    ptdtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type == "cuda" else nullcontext()

    # Parse model tags and task names
    model_tags = args.model_tags.split("|") if args.model_tags else [None]
    task_names = args.task_name.split("|") if args.task_name else []
    for tn in task_names:
        assert tn in TASK_MODULES, f"Unknown task: {tn}. Available: {list(TASK_MODULES.keys())}"

    # Output directory
    output_dir = Path(get_base_dir()) / "plots"
    output_dir.mkdir(exist_ok=True)

    # Process models sequentially to avoid holding all in GPU memory
    token_bytes = get_token_bytes(device=device) if args.val_loss else None
    task_objects = {tn: TASK_MODULES[tn]() for tn in task_names}

    all_bpb_results: dict[str, dict] = {}
    all_task_results: dict[str, dict[str, dict]] = {tn: {} for tn in task_names}

    for model_tag in model_tags:
        tag_label = model_tag or "default"
        print0(f"\nLoading model: {tag_label} (source={args.source})")
        model, tokenizer, _meta = load_model(args.source, device, phase="eval", model_tag=model_tag)
        scaling_params = model.num_scaling_params()
        num_recur = args.num_recur if args.num_recur is not None else int(model.config.train_recur_mean)
        print0(f"  num_recur={num_recur}, params={scaling_params}")

        # --- Val loss (BPB) evaluation ---
        if args.val_loss:
            val_data = "SFT" if args.source == "sft" else "FineWeb"
            print0(f"\n{'=' * 60}")
            print0(f"Val Loss (BPB) on {val_data} — {tag_label} (r={num_recur})")
            print0(f"{'=' * 60}")

            if args.source == "sft":
                val_loader = sft_val_data_generator(
                    tokenizer, args.batch_size, model.config.sequence_len, device,
                )
            else:
                val_loader = tokenizing_distributed_data_loader_bos_bestfit(
                    tokenizer, args.batch_size, model.config.sequence_len, "val", device=device,
                )

            with autocast_ctx:
                bpb_result = evaluate_bpb_with_gates(
                    model, val_loader, token_bytes,
                    batch_size=args.batch_size,
                    num_recur=num_recur,
                    val_tokens=args.val_tokens,
                )

            print0(f"Results for {tag_label}:")
            print0(f"  max_r bpb:   {bpb_result['max_r_bpb']:.4f}")
            print0(f"  train_r bpb: {bpb_result['train_r_bpb']:.4f} [train_r={bpb_result['train_recur']}]")
            print0(f"  r=1 bpb:     {bpb_result['r1_bpb']:.4f}")

            all_bpb_results[tag_label] = {
                "bpb_result": bpb_result,
                "scaling_params": scaling_params,
            }

            gate_train_recur = bpb_result["train_recur"]
            for gate_name, thresholds_dict in bpb_result["gate_results"].items():
                print0(f"\n  {gate_name}:")
                for thr, stats in sorted(thresholds_dict.items()):
                    if stats["total_tokens"] == 0:
                        continue
                    bpb = _bpb_from_nats_bytes(stats["total_nats"], stats["total_bytes"])
                    mean_depth = stats["total_depth"] / stats["total_tokens"]
                    var_depth = stats["total_depth_sq"] / stats["total_tokens"] - mean_depth ** 2
                    std_depth = max(var_depth, 0.0) ** 0.5
                    flops_frac = compute_flops_fraction(scaling_params, mean_depth, gate_train_recur)
                    print0(f"    thr={thr:<8} bpb={bpb:.4f}  mean_depth={mean_depth:.2f}  std={std_depth:.2f}  flops={flops_frac:.4f}")

        # --- Categorical task evaluation ---
        for task_name in task_names:
            print0(f"\n{'=' * 60}")
            print0(f"Task: {task_name} — {tag_label} (r={num_recur})")
            print0(f"{'=' * 60}")

            with autocast_ctx:
                eval_result = evaluate_with_gates(
                    model, tokenizer, task_objects[task_name],
                    batch_size=args.batch_size,
                    num_recur=num_recur,
                    max_problems=args.max_problems,
                )

            total = eval_result["total"]
            print0(f"Results for {tag_label}:")
            print0(f"  max_r accuracy:   {eval_result['max_r_correct']}/{total} ({100 * eval_result['max_r_correct'] / total:.2f}%)")
            print0(f"  train_r accuracy: {eval_result['train_r_correct']}/{total} ({100 * eval_result['train_r_correct'] / total:.2f}%) [train_r={eval_result['train_recur']}]")
            print0(f"  r=1 accuracy:     {eval_result['r1_correct']}/{total} ({100 * eval_result['r1_correct'] / total:.2f}%)")

            all_task_results[task_name][tag_label] = {
                "eval_result": eval_result,
                "scaling_params": scaling_params,
            }

            gate_train_recur = eval_result["train_recur"]
            for gate_name, thresholds_dict in eval_result["gate_results"].items():
                print0(f"\n  {gate_name}:")
                for thr, stats in sorted(thresholds_dict.items()):
                    if stats["total"] == 0:
                        continue
                    acc = stats["correct"] / stats["total"]
                    mean_depth = stats["total_depth"] / stats["total"]
                    flops_frac = compute_flops_fraction(scaling_params, mean_depth, gate_train_recur)
                    print0(f"    thr={thr:<8} acc={acc:.4f}  mean_depth={mean_depth:.2f}  flops={flops_frac:.4f}")

        del model
        print0(f"  Freed model: {tag_label}")

    # --- Plot and save after all models processed ---
    suffix = f"_r{args.num_recur}" if args.num_recur is not None else "_r-native"
    if args.val_loss:
        plot_pareto_bpb(all_bpb_results, output_dir, suffix)
        plot_depth_histograms(all_bpb_results, output_dir, suffix)
        save_bpb_csv(all_bpb_results, output_dir, suffix)

    for task_name in task_names:
        plot_pareto_curves(all_task_results[task_name], task_name, output_dir, suffix)
        save_results_csv(all_task_results[task_name], task_name, output_dir, suffix)

    compute_cleanup()


if __name__ == "__main__":
    main()
