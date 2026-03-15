"""
Unified evaluation script for base models.

Supports three evaluation modes (comma-separated):
  --eval core    : CORE metric (accuracy on ICL tasks)
  --eval bpb     : Bits per byte on train/val splits
  --eval sample  : Generate samples from the model

Default is all three: --eval core,bpb,sample

Examples:

    # Evaluate a HuggingFace model (e.g. GPT-2 124M) using 8 GPUs
    torchrun --nproc_per_node=8 -m scripts.base_eval --hf-path openai-community/gpt2

    # Evaluate a nanochat model (e.g. d24) using 8 GPUs
    torchrun --nproc_per_node=8 -m scripts.base_eval --model-tag d24 --device-batch-size=16

    # Quick/approximate evaluation using a single GPU
    python -m scripts.base_eval --model-tag d24 --device-batch-size=16 --max-per-task=100 --split-tokens=524288
"""

import argparse
import csv
import json
import os
import random
import shutil
import tempfile
import time
import zipfile
from contextlib import nullcontext
from pathlib import Path

import torch
import yaml

from nanochat.checkpoint_manager import load_model
from nanochat.common import autodetect_device_type, compute_cleanup, compute_init, download_file_with_lock, get_base_dir, print0
from nanochat.core_eval import evaluate_task
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
from nanochat.engine import Engine
from nanochat.loss_eval import evaluate_bpb
from nanochat.tokenizer import HuggingFaceTokenizer, get_token_bytes

# -----------------------------------------------------------------------------
# HuggingFace loading utilities


class ModelWrapper:
    """Lightweight wrapper to give HuggingFace models a nanochat-compatible interface."""

    def __init__(self, model, max_seq_len=None):
        self.model = model
        self.max_seq_len = max_seq_len

    def __call__(self, input_ids, targets=None, loss_reduction="mean"):
        logits = self.model(input_ids).logits
        if targets is None:
            return logits
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction=loss_reduction
        )
        return loss

    def get_device(self):
        return next(self.model.parameters()).device


def load_hf_model(hf_path: str, device):
    """Load a HuggingFace model and tokenizer."""
    print0(f"Loading HuggingFace model from: {hf_path}")
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(hf_path)
    model.to(device)
    model.eval()
    max_seq_len = 1024 if "openai-community/gpt2" in hf_path else None
    model = ModelWrapper(model, max_seq_len=max_seq_len)
    tokenizer = HuggingFaceTokenizer.from_pretrained(hf_path)
    return model, tokenizer


def get_hf_token_bytes(tokenizer, device="cpu"):
    """Compute token_bytes tensor for a HuggingFace tokenizer."""
    vocab_size = tokenizer.tokenizer.get_vocab_size()
    token_bytes = torch.zeros(vocab_size, dtype=torch.int64, device=device)
    for token_id in range(vocab_size):
        token_str = tokenizer.tokenizer.decode([token_id])
        token_bytes[token_id] = len(token_str.encode("utf-8"))
    return token_bytes


# -----------------------------------------------------------------------------
# CORE evaluation

EVAL_BUNDLE_URL = "https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip"


def place_eval_bundle(file_path):
    """Unzip eval_bundle.zip and place it in the base directory."""
    base_dir = get_base_dir()
    eval_bundle_dir = os.path.join(base_dir, "eval_bundle")
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(file_path, "r") as zip_ref:
            zip_ref.extractall(tmpdir)
        extracted_bundle_dir = os.path.join(tmpdir, "eval_bundle")
        shutil.move(extracted_bundle_dir, eval_bundle_dir)
    print0(f"Placed eval_bundle directory at {eval_bundle_dir}")


def evaluate_core(model, tokenizer, device, max_per_task=-1, num_recur=None):
    """
    Evaluate a base model on the CORE benchmark.
    Returns dict with results, centered_results, and core_metric.
    """
    base_dir = get_base_dir()
    eval_bundle_dir = os.path.join(base_dir, "eval_bundle")
    # Download the eval bundle if needed
    if not os.path.exists(eval_bundle_dir):
        download_file_with_lock(EVAL_BUNDLE_URL, "eval_bundle.zip", postprocess_fn=place_eval_bundle)

    config_path = os.path.join(eval_bundle_dir, "core.yaml")
    data_base_path = os.path.join(eval_bundle_dir, "eval_data")
    eval_meta_data = os.path.join(eval_bundle_dir, "eval_meta_data.csv")

    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    tasks = config["icl_tasks"]

    # Load random baseline values
    random_baselines = {}
    with open(eval_meta_data, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            task_name = row["Eval Task"]
            random_baseline = row["Random baseline"]
            random_baselines[task_name] = float(random_baseline)

    # Evaluate each task
    results = {}
    centered_results = {}
    task_groups = {}  # label -> group name (derived from dataset_uri directory)
    for task in tasks:
        start_time = time.time()
        label = task["label"]
        group = task["dataset_uri"].split("/")[0]
        task_groups[label] = group
        task_meta = {
            "task_type": task["icl_task_type"],
            "dataset_uri": task["dataset_uri"],
            "num_fewshot": task["num_fewshot"][0],
            "continuation_delimiter": task.get("continuation_delimiter", " "),
        }
        print0(f"Evaluating: {label} ({task_meta['num_fewshot']}-shot, type: {task_meta['task_type']})... ", end="")

        data_path = os.path.join(data_base_path, task_meta["dataset_uri"])
        with open(data_path, encoding="utf-8") as f:
            data = [json.loads(line.strip()) for line in f]

        # Shuffle for consistent subsampling when using max_per_task
        shuffle_rng = random.Random(1337)
        shuffle_rng.shuffle(data)
        if max_per_task > 0:
            data = data[:max_per_task]

        accuracy = evaluate_task(model, tokenizer, data, device, task_meta, num_recur=num_recur)
        results[label] = accuracy
        random_baseline = random_baselines[label]
        centered_result = (accuracy - 0.01 * random_baseline) / (1.0 - 0.01 * random_baseline)
        centered_results[label] = centered_result
        elapsed = time.time() - start_time
        print0(f"accuracy: {accuracy:.4f} | centered: {centered_result:.4f} | time: {elapsed:.2f}s")

    # Group averages (centered scores)
    group_labels: dict[str, list[str]] = {}
    for label, group in task_groups.items():
        group_labels.setdefault(group, []).append(label)
    group_results = {g: sum(centered_results[l] for l in labels) / len(labels)
                     for g, labels in group_labels.items()}

    core_metric = sum(centered_results.values()) / len(centered_results)
    out = {"results": results, "centered_results": centered_results,
           "group_results": group_results, "core_metric": core_metric}
    return out


# -----------------------------------------------------------------------------
# Saunshi evaluation


SAUNSHI_BUNDLE_DIR = Path(get_base_dir()) / "saunshi_bundle"


def evaluate_saunshi(model, tokenizer, device, max_per_task=-1, num_recur=None):
    """
    Evaluate a base model on the Saunshi math word problem benchmark.
    Returns dict with results (per-task accuracy), group_results (per-group average),
    and saunshi_metric (mean of group averages).
    """
    bundle_dir = SAUNSHI_BUNDLE_DIR
    config_path = bundle_dir / "saunshi.yaml"
    data_base_path = bundle_dir / "eval_data"

    if not config_path.exists():
        raise FileNotFoundError(
            f"Saunshi bundle not found at {bundle_dir}. "
            "Run: uv run python dev/eval_saunshi/prepare_saunshi_bundle.py"
        )

    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    tasks = config["tasks"]

    results = {}
    for task in tasks:
        start_time = time.time()
        label = task["label"]
        task_meta = {
            "task_type": task["task_type"],
            "dataset_uri": task["dataset_uri"],
            "num_fewshot": task["num_fewshot"],
            "continuation_delimiter": task.get("continuation_delimiter", " "),
        }
        print0(f"Evaluating: {label} ({task_meta['num_fewshot']}-shot, type: {task_meta['task_type']})... ", end="")

        data_path = data_base_path / task_meta["dataset_uri"]
        with open(data_path, encoding="utf-8") as f:
            data = [json.loads(line.strip()) for line in f]

        shuffle_rng = random.Random(1337)
        shuffle_rng.shuffle(data)
        if max_per_task > 0:
            data = data[:max_per_task]

        accuracy = evaluate_task(model, tokenizer, data, device, task_meta, num_recur=num_recur)
        results[label] = accuracy
        elapsed = time.time() - start_time
        print0(f"accuracy: {accuracy:.4f} | time: {elapsed:.2f}s")

    # Compute per-group averages
    group_tasks: dict[str, list[str]] = {}
    for task in tasks:
        g = task.get("group", "ungrouped")
        group_tasks.setdefault(g, []).append(task["label"])

    group_results = {}
    for group, labels in group_tasks.items():
        group_results[group] = sum(results[l] for l in labels) / len(labels)

    saunshi_metric = sum(group_results.values()) / len(group_results)
    return {"results": results, "group_results": group_results, "saunshi_metric": saunshi_metric}


# -----------------------------------------------------------------------------
# Main


def main():
    parser = argparse.ArgumentParser(description="Base model evaluation")
    parser.add_argument(
        "--eval", type=str, default="core,bpb,sample", help="Comma-separated evaluations to run: core,bpb,sample (default: core,bpb,sample)"
    )
    parser.add_argument("--hf-path", type=str, default=None, help="HuggingFace model path (e.g. openai-community/gpt2)")
    parser.add_argument("--model-tag", type=str, default=None, help="nanochat model tag to identify the checkpoint directory")
    parser.add_argument("--step", type=int, default=None, help="Model step to load (default = last)")
    parser.add_argument("--max-per-task", type=int, default=-1, help="Max examples per CORE task (-1 = all)")
    parser.add_argument("--device-batch-size", type=int, default=32, help="Per-device batch size for BPB evaluation")
    parser.add_argument("--split-tokens", type=int, default=40 * 524288, help="Number of tokens to evaluate per split for BPB")
    parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")
    parser.add_argument("--num-recur", type=str, default=None, help="Comma-separated recurrence depths to evaluate, e.g. '2,4,6' (default = model's train_recur_mean)")

    args = parser.parse_args()

    # Parse evaluation modes
    eval_modes = set(mode.strip() for mode in args.eval.split(","))
    valid_modes = {"core", "bpb", "sample", "saunshi"}
    invalid = eval_modes - valid_modes
    if invalid:
        parser.error(f"Invalid eval modes: {invalid}. Valid: {valid_modes}")

    # Distributed / precision setup
    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()

    # Load model and tokenizer
    is_hf_model = args.hf_path is not None
    if is_hf_model:
        model, tokenizer = load_hf_model(args.hf_path, device)
        sequence_len = model.max_seq_len or 1024
        token_bytes = get_hf_token_bytes(tokenizer, device=device)
    else:
        model, tokenizer, meta = load_model("base", device, phase="eval", model_tag=args.model_tag, step=args.step)
        sequence_len = meta["model_config"]["sequence_len"]
        token_bytes = get_token_bytes(device=device)

    # Parse num_recur into a list of values to sweep
    if args.num_recur is not None:
        num_recur_values = [int(x) for x in args.num_recur.split(",")]
    elif is_hf_model:
        num_recur_values = [None]
    else:
        num_recur_values = [meta["train_config"].get("train_recur_mean")]

    print0(f"Eval modes: {', '.join(sorted(eval_modes))}")
    print0(f"Recursion depths to evaluate: {num_recur_values}")

    # Base slug for combined CSV output (without recur suffix)
    if is_hf_model:
        base_slug = args.hf_path.replace("/", "-")
    else:
        base_slug = f"{args.model_tag}_step{meta['step']:06d}"

    # Summary CSV: one row per num_recur with aggregate metrics
    base_dir = get_base_dir()
    eval_dir = os.path.join(base_dir, "base_eval")
    os.makedirs(eval_dir, exist_ok=True)
    summary_csv_path = os.path.join(eval_dir, f"{base_slug}.csv")

    def read_completed_recur(csv_path: str) -> set[str]:
        """Read a CSV and return the set of num_recur values already present."""
        if not os.path.exists(csv_path):
            return set()
        completed = set()
        with open(csv_path, encoding="utf-8") as f:
            for line in f:
                if line.startswith("#") or line.startswith("num_recur"):
                    continue
                parts = line.split(",", 1)
                if parts:
                    completed.add(parts[0].strip())
        return completed

    completed_recur = read_completed_recur(summary_csv_path)

    # Write comment header with hyperparameters (once, when creating the file)
    if ddp_rank == 0 and not os.path.exists(summary_csv_path):
        model_id = args.hf_path if is_hf_model else f"{args.model_tag}_step{meta['step']}"
        with open(summary_csv_path, "w", encoding="utf-8", newline="") as f:
            f.write(f"# model={model_id}, eval={args.eval}"
                    f", device_batch_size={args.device_batch_size}"
                    f", split_tokens={args.split_tokens}"
                    f", max_per_task={args.max_per_task}\n")

    for num_recur in num_recur_values:
        nr_str = str(num_recur)

        if nr_str in completed_recur:
            print0(f"\nSkipping num_recur={num_recur} (already in CSV)")
            continue

        # Build model name for this recursion depth
        if is_hf_model:
            model_name = args.hf_path
        else:
            model_name = f"{args.model_tag} (step {meta['step']}, num_recur={num_recur})"

        print0(f"\n{'#' * 80}")
        print0(f"Evaluating model: {model_name}")
        print0(f"{'#' * 80}")

        # Results to log
        core_results = None
        saunshi_results = None
        bpb_results = {}

        # --- CORE evaluation ---
        if "core" in eval_modes:
            print0("\n" + "=" * 80)
            print0("CORE Evaluation")
            print0("=" * 80)
            with autocast_ctx:
                core_results = evaluate_core(model, tokenizer, device, max_per_task=args.max_per_task, num_recur=num_recur)
            print0("-" * 80)
            for group, acc in core_results["group_results"].items():
                print0(f"  {group:<30s} {acc:.4f}")
            print0("-" * 80)
            print0(f"  {'core_metric':<30s} {core_results['core_metric']:.4f}")

        # --- Saunshi evaluation ---
        if "saunshi" in eval_modes:
            print0("\n" + "=" * 80)
            print0("Saunshi Evaluation")
            print0("=" * 80)
            with autocast_ctx:
                saunshi_results = evaluate_saunshi(model, tokenizer, device, max_per_task=args.max_per_task, num_recur=num_recur)
            print0("-" * 80)
            for group, acc in saunshi_results["group_results"].items():
                print0(f"  {group:<30s} {acc:.4f}")
            print0("-" * 80)
            print0(f"  {'saunshi_metric':<30s} {saunshi_results['saunshi_metric']:.4f}")

        # --- BPB evaluation ---
        if "bpb" in eval_modes:
            print0("\n" + "=" * 80)
            print0("BPB Evaluation")
            print0("=" * 80)
            tokens_per_step = args.device_batch_size * sequence_len * ddp_world_size
            if args.split_tokens % tokens_per_step != 0:
                args.split_tokens = (args.split_tokens // tokens_per_step) * tokens_per_step
                print0(f"Adjusted split_tokens to {args.split_tokens} (must be divisible by {tokens_per_step})")
            steps = args.split_tokens // tokens_per_step

            for split_name in ["train", "val"]:
                loader = tokenizing_distributed_data_loader_bos_bestfit(
                    tokenizer, args.device_batch_size, sequence_len, split_name, device=device
                )
                with autocast_ctx:
                    bpb = evaluate_bpb(model, loader, steps, token_bytes, num_recur=num_recur)
                bpb_results[split_name] = bpb
                print0(f"{split_name} bpb: {bpb:.6f}")

        # --- Append summary row to CSV ---
        if ddp_rank == 0:
            # Build header and row dynamically based on what was evaluated
            columns = ["num_recur"]
            values = [nr_str]
            if core_results:
                columns.append("core_metric")
                values.append(f"{core_results['core_metric']:.6f}")
                for label in core_results["centered_results"]:
                    columns.append(label)
                    values.append(f"{core_results['centered_results'][label]:.6f}")
            if saunshi_results:
                columns.append("saunshi_metric")
                values.append(f"{saunshi_results['saunshi_metric']:.6f}")
                for group, acc in saunshi_results["group_results"].items():
                    columns.append(f"saunshi_{group}")
                    values.append(f"{acc:.6f}")
                for label, acc in saunshi_results["results"].items():
                    columns.append(f"saunshi_{label}")
                    values.append(f"{acc:.6f}")
            if bpb_results:
                columns.append("train_bpb")
                values.append(f"{bpb_results['train']:.6f}")
                columns.append("val_bpb")
                values.append(f"{bpb_results['val']:.6f}")

            # Write header row if this is the first data row
            file_size = os.path.getsize(summary_csv_path) if os.path.exists(summary_csv_path) else 0
            write_header = file_size == 0 or all(
                line.startswith("#") for line in open(summary_csv_path, encoding="utf-8") if line.strip()
            )
            with open(summary_csv_path, "a", encoding="utf-8", newline="") as f:
                if write_header:
                    f.write(",".join(columns) + "\n")
                f.write(",".join(values) + "\n")
            print0(f"\nSummary appended to: {summary_csv_path}")

        # --- Sampling ---
        if "sample" in eval_modes and not is_hf_model:
            print0("\n" + "=" * 80)
            print0("Model Samples")
            print0("=" * 80)
            if ddp_rank == 0:
                prompts = [
                    "The capital of France is",
                    "The chemical symbol of gold is",
                    "If yesterday was Friday, then tomorrow will be",
                    "The opposite of hot is",
                    "The planets of the solar system are:",
                    "My favorite color is",
                    "If 5*x + 3 = 13, then x is",
                ]
                engine = Engine(model, tokenizer)
                print0("\nConditioned samples:")
                for prompt in prompts:
                    tokens = tokenizer(prompt, prepend="<|bos|>")
                    with autocast_ctx:
                        sample, _ = engine.generate_batch(tokens, num_samples=1, max_tokens=16, temperature=0, num_recur=num_recur)
                    sample_str = tokenizer.decode(sample[0])
                    print0("-" * 80)
                    print0(sample_str)

                print0("\nUnconditioned samples:")
                tokens = tokenizer("", prepend="<|bos|>")
                with autocast_ctx:
                    uncond, _ = engine.generate_batch(tokens, num_samples=8, max_tokens=128, temperature=1.0, num_recur=num_recur)
                for sample in uncond:
                    sample_str = tokenizer.decode(sample)
                    print0("-" * 80)
                    print0(sample_str)
        elif "sample" in eval_modes and is_hf_model:
            print0("\nSkipping sampling for HuggingFace models (not supported)")

    compute_cleanup()


if __name__ == "__main__":
    main()
