"""
Plot the 1-sqrt recurrence curriculum schedule.

Usage:
    uv run python -m dev.analysis.plot_recur_schedule --target-mean 4 --warmup-ratio 0.6
    uv run python -m dev.analysis.plot_recur_schedule --target-mean 32 --warmup-ratio 0.5 --num-iterations 50000
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from nanochat.common import get_base_dir, get_scheduled_recur_mean


def plot_schedule(target_mean: float, warmup_ratio: float, num_iterations: int):
    """Plot scheduled recurrence mean vs training step."""
    steps = np.arange(num_iterations + 1)
    means = [get_scheduled_recur_mean(int(s), num_iterations, target_mean, warmup_ratio) for s in steps]
    warmup_end = round(warmup_ratio * num_iterations)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(steps, means, linewidth=2, color="steelblue")
    ax.axhline(target_mean, color="darkgreen", linestyle="--", linewidth=1.5, label=f"target = {target_mean}")
    ax.axvline(warmup_end, color="red", linestyle=":", linewidth=1.5, label=f"warmup end (step {warmup_end})")
    ax.set_xlabel("Training step", fontsize=12)
    ax.set_ylabel("Scheduled recurrence mean", fontsize=12)
    ax.set_title(
        f"1-sqrt Recurrence Curriculum\n(target={target_mean}, warmup_ratio={warmup_ratio}, N={num_iterations})",
        fontsize=14,
    )
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()

    plots_dir = Path(get_base_dir()) / "plots"
    plots_dir.mkdir(exist_ok=True)
    filename = f"recur_schedule_target{target_mean}_warmup{warmup_ratio}_N{num_iterations}.png"
    save_path = plots_dir / filename
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot 1-sqrt recurrence curriculum schedule")
    parser.add_argument("--target-mean", type=float, default=4.0, help="Target recurrence mean (default: 4.0)")
    parser.add_argument("--warmup-ratio", type=float, default=0.5, help="Fraction of training for ramp (default: 0.5)")
    parser.add_argument("--num-iterations", type=int, default=10000, help="Total training steps (default: 10000)")
    args = parser.parse_args()
    plot_schedule(args.target_mean, args.warmup_ratio, args.num_iterations)


if __name__ == "__main__":
    main()
