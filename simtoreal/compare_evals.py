"""
Compare evaluation results across multiple runs.

Usage:
    python compare_evals.py \
        --results ../runs/compare_cqn_as/eval_results.json \
                  ../runs/compare_arsq_base/eval_results.json \
                  ../runs/compare_arsq_phys/eval_results.json \
        --labels "CQN-AS" "ARSQ-base" "ARSQ+Physics" \
        --save eval_comparison.png

Each eval_results.json is produced by eval_real.py and contains:
  [{"episode": 1, "steps": 200, "reward": 0.0, "success": false}, ...]
"""

import argparse
import json
import sys
import matplotlib.pyplot as plt
import numpy as np


def load_results(path: str) -> list[dict]:
    with open(path, "r") as f:
        return json.load(f)


def print_table(all_results: list[list[dict]], labels: list[str]):
    """Print a comparison table to stdout."""
    print("\n" + "=" * 72)
    print("EVALUATION COMPARISON")
    print("=" * 72)

    header = f"{'Metric':<25}"
    for label in labels:
        header += f"{label:>15}"
    print(header)
    print("-" * 72)

    for metric_name, extract_fn, fmt in [
        ("Episodes", lambda rs: len(rs), "d"),
        ("Successes", lambda rs: sum(1 for r in rs if r["success"]), "d"),
        ("Success Rate", lambda rs: sum(1 for r in rs if r["success"]) / max(len(rs), 1), ".1%"),
        ("Mean Reward", lambda rs: np.mean([r["reward"] for r in rs]), ".3f"),
        ("Mean Length", lambda rs: np.mean([r["steps"] for r in rs]), ".1f"),
    ]:
        row = f"{metric_name:<25}"
        for results in all_results:
            if not results:
                row += f"{'N/A':>15}"
            else:
                val = extract_fn(results)
                row += f"{val:>15{fmt}}"
        print(row)
    print("=" * 72)


def plot_comparison(all_results: list[list[dict]], labels: list[str],
                    save_path: str | None = None):
    COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # ── Bar chart: Success Rate ──
    success_rates = []
    for results in all_results:
        if results:
            sr = sum(1 for r in results if r["success"]) / len(results) * 100
        else:
            sr = 0
        success_rates.append(sr)

    bars = axes[0].bar(labels, success_rates, color=COLORS[:len(labels)], alpha=0.8)
    axes[0].set_ylabel("Success Rate (%)")
    axes[0].set_title("Success Rate")
    axes[0].set_ylim(0, 105)
    for bar, val in zip(bars, success_rates):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                     f"{val:.0f}%", ha="center", va="bottom", fontweight="bold")

    # ── Bar chart: Mean Reward ──
    mean_rewards = []
    for results in all_results:
        if results:
            mean_rewards.append(np.mean([r["reward"] for r in results]))
        else:
            mean_rewards.append(0)

    bars = axes[1].bar(labels, mean_rewards, color=COLORS[:len(labels)], alpha=0.8)
    axes[1].set_ylabel("Mean Reward")
    axes[1].set_title("Mean Reward")
    for bar, val in zip(bars, mean_rewards):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f"{val:.2f}", ha="center", va="bottom", fontweight="bold")

    # ── Per-episode success (dot plot) ──
    for i, (results, label) in enumerate(zip(all_results, labels)):
        if not results:
            continue
        episodes = [r["episode"] for r in results]
        successes = [1 if r["success"] else 0 for r in results]
        axes[2].scatter(
            episodes, [i] * len(episodes),
            c=[COLORS[i] if s else "#cccccc" for s in successes],
            s=80, marker="s", edgecolors="black", linewidths=0.5,
        )
    axes[2].set_yticks(range(len(labels)))
    axes[2].set_yticklabels(labels)
    axes[2].set_xlabel("Episode")
    axes[2].set_title("Per-Episode Success (colored = success)")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\nSaved comparison plot to {save_path}")
    else:
        plt.show()


def main():
    p = argparse.ArgumentParser(description="Compare eval results across runs")
    p.add_argument("--results", nargs="+", required=True,
                   help="Paths to eval_results.json files")
    p.add_argument("--labels", nargs="+",
                   help="Labels for each run (default: parent folder names)")
    p.add_argument("--save", help="Save plot to file instead of showing")
    args = p.parse_args()

    if args.labels is None:
        from pathlib import Path
        args.labels = [Path(p).parent.name for p in args.results]

    if len(args.labels) != len(args.results):
        print("Error: number of --labels must match number of --results",
              file=sys.stderr)
        sys.exit(1)

    all_results = []
    for path in args.results:
        try:
            results = load_results(path)
            all_results.append(results)
            n_success = sum(1 for r in results if r["success"])
            print(f"Loaded {path}: {len(results)} episodes, {n_success} successes")
        except FileNotFoundError:
            print(f"Warning: {path} not found, skipping", file=sys.stderr)
            all_results.append([])

    print_table(all_results, args.labels)
    plot_comparison(all_results, args.labels, args.save)


if __name__ == "__main__":
    main()
