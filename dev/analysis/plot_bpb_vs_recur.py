"""
Plot validation BPB vs number of recurrences for different models.

Reads summary CSV files from base_eval directory (produced by base_eval.py)
and creates a comparison plot showing how validation loss changes with
test-time compute (num_recur).

CSV format (from base_eval.py):
    # model=..., eval=..., ...
    num_recur,core_metric,...,train_bpb,val_bpb
    4,0.123,...,0.85,0.87

Usage:
    uv run python dev/analysis/plot_bpb_vs_recur.py [--step STEP] [--model-tags TAG1 TAG2 ...]
"""

import argparse
import csv
import os
import re
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt

# Source machine_config.sh so NANOCHAT_BASE_DIR is set before get_base_dir()
_config_path = Path(__file__).resolve().parents[2] / "shells" / "machine_config.sh"
if _config_path.exists() and not os.environ.get("NANOCHAT_BASE_DIR"):
    _out = subprocess.run(
        ["bash", "-c", f"source {_config_path} && env"],
        capture_output=True, text=True, check=True,
    )
    for line in _out.stdout.splitlines():
        key, _, val = line.partition("=")
        if key in ("NANOCHAT_BASE_DIR", "HF_HOME", "HF_DATASETS_CACHE"):
            os.environ[key] = val

from nanochat.common import get_base_dir


# Colorblind-safe palette (dark blues/purples + bright accents)
COLORS = [
    "#8cc5e3",  # Bright Mint
    "#0d7d87",  # Bright Cyan
    "#4a2377",  # Deep Navy Blue
    "#f55f74",  # Dusky Purple
    "#FFD166",  # Bright Gold
    "#FF8C42",  # Bright Orange
    "#FF66C4",  # Bright Pink
    "#7EC8E3",  # Bright Sky Blue
]

# Distinct marker styles
MARKERS = ["o", "s", "^", "D", "v", "<", ">", "p", "*", "h", "H", "X"]


def parse_filename(filename: str) -> tuple[str, int] | None:
    """
    Parse filename to extract model_tag and step.

    Expected format: {model_tag}_step{step:06d}.csv
    Returns (model_tag, step) or None if parsing fails.
    """
    pattern = r"^(.+)_step(\d+)\.csv$"
    match = re.match(pattern, filename)
    if match:
        return (match.group(1), int(match.group(2)))
    return None


def parse_summary_csv(csv_path: Path) -> list[dict[str, float]]:
    """
    Parse a base_eval summary CSV file.

    Skips comment lines (starting with #) and reads the header + data rows.
    Returns list of dicts with at least 'num_recur', 'train_bpb', 'val_bpb'.
    Rows missing BPB columns are skipped.
    """
    rows = []
    with open(csv_path, encoding="utf-8") as f:
        # Filter out comment lines before passing to csv reader
        non_comment_lines = [line for line in f if not line.startswith("#")]
    if not non_comment_lines:
        return rows

    reader = csv.DictReader(non_comment_lines)
    for row in reader:
        if "val_bpb" not in row or "num_recur" not in row:
            continue
        try:
            parsed = {
                "num_recur": int(row["num_recur"]) if row["num_recur"] != "None" else 0,
                "val_bpb": float(row["val_bpb"]),
                "train_bpb": float(row["train_bpb"]) if "train_bpb" in row else None,
            }
            rows.append(parsed)
        except (ValueError, KeyError):
            continue
    return rows


def collect_data(base_eval_dir: Path, step_filter: int | None = None, model_tags_filter: list[str] | None = None) -> dict:
    """
    Collect all BPB results from summary CSV files.

    Returns dict: {model_tag: {step: {num_recur: {"train": float, "val": float}}}}
    """
    data: dict = {}

    for csv_file in base_eval_dir.glob("*.csv"):
        parsed = parse_filename(csv_file.name)
        if parsed is None:
            continue

        model_tag, step = parsed

        # Apply filters
        if step_filter is not None and step != step_filter:
            continue
        if model_tags_filter is not None and model_tag not in model_tags_filter:
            continue

        # Parse rows from summary CSV
        rows = parse_summary_csv(csv_file)
        if not rows:
            continue

        if model_tag not in data:
            data[model_tag] = {}
        if step not in data[model_tag]:
            data[model_tag][step] = {}

        for row in rows:
            num_recur = row["num_recur"]
            bpb_values = {"val": row["val_bpb"]}
            if row["train_bpb"] is not None:
                bpb_values["train"] = row["train_bpb"]
            data[model_tag][step][num_recur] = bpb_values

    return data


def parse_mean_recur(model_tag: str) -> int | None:
    """Extract mean training recurrence from model tag prefix (e.g., 'r4_...' -> 4)."""
    match = re.match(r"^r(\d+)_", model_tag)
    if match:
        return int(match.group(1))
    return None


def create_short_label(model_tag: str) -> str:
    """Create a shorter, more readable label from model_tag."""
    # Remove common prefixes/suffixes for cleaner labels
    label = model_tag
    # Example: "r4_sample_init_random_2.15e18_s12" -> "r4_sample_init_random"
    # Remove FLOPs count and seed suffix
    label = re.sub(r"_[\d.]+e\d+_s\d+$", "", label)
    return label


def plot_bpb_vs_recur(data: dict, output_path: Path):
    """Create plot of validation BPB vs num_recur for each model."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Collect all model_tags and steps
    model_step_pairs = []
    for model_tag, steps_dict in sorted(data.items()):
        for step in sorted(steps_dict.keys()):
            model_step_pairs.append((model_tag, step))

    # Plot each model-step combination
    for idx, (model_tag, step) in enumerate(model_step_pairs):
        recur_dict = data[model_tag][step]

        # Extract and sort by num_recur
        recur_vals = sorted(recur_dict.keys())
        val_bpbs = [recur_dict[r]["val"] for r in recur_vals]

        # Create label
        short_label = create_short_label(model_tag)
        if len(data[model_tag]) > 1:  # Multiple steps for this model
            label = f"{short_label} (step {step})"
        else:
            label = short_label

        # Plot with colorblind-safe colors and distinct markers
        color = COLORS[idx % len(COLORS)]
        marker = MARKERS[idx % len(MARKERS)]

        ax.plot(
            recur_vals,
            val_bpbs,
            marker=marker,
            markersize=8,
            linewidth=2,
            linestyle="-",
            color=color,
            label=label,
        )

    # Add visual marks for specific recurrence ranges
    # Shaded region for num_recur=2-16
    ax.axvspan(2, 16, alpha=0.15, color="gray", zorder=0, label="sample range")

    # Vertical lines for each unique mean training recurrence
    seen_recur: set[int] = set()
    for model_tag in data:
        mean_recur = parse_mean_recur(model_tag)
        if mean_recur is not None and mean_recur not in seen_recur:
            seen_recur.add(mean_recur)
            ax.axvline(x=mean_recur, color="darkred", linestyle="--", linewidth=2, alpha=0.7, zorder=1, label=f"mean train recur={mean_recur}")

    # Styling
    ax.set_xlabel("Number of Recurrences", fontsize=12, fontweight="bold")
    ax.set_ylabel("Validation BPB", fontsize=12, fontweight="bold")
    ax.set_title("Test-Time Compute Scaling: Validation BPB vs Recurrence Depth", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="best", framealpha=0.9)

    # Set x-axis to show integer recurrence values
    if len(model_step_pairs) > 0:
        all_recur_vals = set()
        for model_tag, step in model_step_pairs:
            all_recur_vals.update(data[model_tag][step].keys())
        ax.set_xticks(sorted(all_recur_vals))

    plt.tight_layout()

    # Save plot
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to: {output_path}")

    # Also save as PDF for papers
    pdf_path = output_path.with_suffix(".pdf")
    plt.savefig(pdf_path, bbox_inches="tight")
    print(f"PDF saved to: {pdf_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot validation BPB vs number of recurrences")
    parser.add_argument("--step", type=int, default=None, help="Filter by specific checkpoint step")
    parser.add_argument("--model-tags", nargs="+", default=None, help="Filter by specific model tags")
    parser.add_argument("--output", type=str, default=None, help="Output path (default: plots/bpb_vs_recur.png)")
    args = parser.parse_args()

    # Get base directory
    base_dir = Path(get_base_dir())
    base_eval_dir = base_dir / "base_eval"

    if not base_eval_dir.exists():
        print(f"Error: base_eval directory not found at {base_eval_dir}")
        return

    # Collect data
    print("Collecting BPB results from CSV files...")
    data = collect_data(base_eval_dir, step_filter=args.step, model_tags_filter=args.model_tags)

    if not data:
        print("No BPB CSV files found matching the filters.")
        print(f"Looking in: {base_eval_dir}")
        if args.step:
            print(f"Step filter: {args.step}")
        if args.model_tags:
            print(f"Model tags filter: {args.model_tags}")
        return

    # Print summary
    print(f"\nFound results for {len(data)} model(s):")
    for model_tag, steps_dict in sorted(data.items()):
        for step, recur_dict in sorted(steps_dict.items()):
            recur_vals = sorted(recur_dict.keys())
            print(f"  {model_tag} (step {step}): {len(recur_vals)} recurrence values {recur_vals}")

    # Create plot
    output_path = Path(args.output) if args.output else (base_dir / "plots" / "bpb_vs_recur.png")
    print(f"\nCreating plot...")
    plot_bpb_vs_recur(data, output_path)


if __name__ == "__main__":
    main()
