"""
Compare training curves across multiple runs.

Usage:
    python plot_comparison.py \
        --logs run1/train.log run2/train.log run3/train.log \
        --labels "CQN-AS" "ARSQ-base" "ARSQ+Physics" \
        --save comparison.png

Expects stdout logs produced by train_real.py piped through tee:
    python train_real.py ... 2>&1 | tee train.log
"""

import argparse
import re
import sys
import matplotlib.pyplot as plt
import numpy as np


# ── Log Parsing (same patterns as plot_training.py) ──────────────

EP_PAT = re.compile(
    r"Episode\s+(\d+):\s+reward=([\d\.\-]+),\s+length=(\d+),.*step=(\d+)"
)
SUCCESS_PAT = re.compile(
    r"\[([✓✗])\s*AutoReward\]\s+pos_err=([\d\.]+)cm.*quat_err=([\d\.]+)"
)
SIMPLE_SUCCESS = re.compile(r"Success! reward=1\.0")
SIMPLE_FAIL = re.compile(r"No success\.")


def parse_log(path: str) -> list[dict]:
    """Parse a training log file and return a list of episode dicts."""
    episodes = []
    with open(path, "r") as f:
        lines = f.readlines()

    pending = None
    for line in lines:
        line = line.strip()

        m = EP_PAT.search(line)
        if m:
            pending = {
                "episode": int(m.group(1)),
                "reward": float(m.group(2)),
                "length": int(m.group(3)),
                "step": int(m.group(4)),
                "success": None,
                "pos_err": None,
            }
            episodes.append(pending)
            continue

        m = SUCCESS_PAT.search(line)
        if m and pending is not None:
            pending["success"] = 1 if m.group(1) == "✓" else 0
            pending["pos_err"] = float(m.group(2))
            continue

        if SIMPLE_SUCCESS.search(line) and pending is not None:
            if pending["success"] is None:
                pending["success"] = 1
        elif SIMPLE_FAIL.search(line) and pending is not None:
            if pending["success"] is None:
                pending["success"] = 0

    return episodes


def smooth(vals, window: int):
    if len(vals) < window:
        return vals, list(range(len(vals)))
    kernel = np.ones(window) / window
    return np.convolve(vals, kernel, mode="valid"), list(range(window - 1, len(vals)))


# ── Plotting ─────────────────────────────────────────────────────

COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]


def plot_comparison(all_episodes: list[list[dict]], labels: list[str],
                    save_path: str | None = None, window: int = 10):
    # Determine which metrics are available
    has_success = any(
        e["success"] is not None
        for eps in all_episodes for e in eps
    )
    has_pos_err = any(
        e["pos_err"] is not None
        for eps in all_episodes for e in eps
    )

    n_plots = 2 + int(has_success) + int(has_pos_err)
    fig, axes = plt.subplots(n_plots, 1, figsize=(12, 3.5 * n_plots), sharex=True)
    if n_plots == 1:
        axes = [axes]

    for run_idx, (episodes, label) in enumerate(zip(all_episodes, labels)):
        if not episodes:
            print(f"Warning: no episodes found for '{label}'", file=sys.stderr)
            continue

        color = COLORS[run_idx % len(COLORS)]
        steps = [e["step"] for e in episodes]
        rewards = [e["reward"] for e in episodes]
        lengths = [e["length"] for e in episodes]

        ax_idx = 0

        # ── Episode Reward ──
        axes[ax_idx].plot(steps, rewards, color=color, alpha=0.15, linewidth=0.7)
        if len(rewards) >= window:
            sm, sm_idx = smooth(rewards, window)
            sm_steps = [steps[i] for i in sm_idx]
            axes[ax_idx].plot(sm_steps, sm, color=color, linewidth=2.2, label=label)
        else:
            axes[ax_idx].plot(steps, rewards, color=color, linewidth=1.5, label=label)
        ax_idx += 1

        # ── Episode Length ──
        axes[ax_idx].plot(steps, lengths, color=color, alpha=0.15, linewidth=0.7)
        if len(lengths) >= window:
            sm, sm_idx = smooth(lengths, window)
            sm_steps = [steps[i] for i in sm_idx]
            axes[ax_idx].plot(sm_steps, sm, color=color, linewidth=2.2, label=label)
        else:
            axes[ax_idx].plot(steps, lengths, color=color, linewidth=1.5, label=label)
        ax_idx += 1

        # ── Success Rate ──
        if has_success:
            successes = [
                e["success"] if e["success"] is not None else 0
                for e in episodes
            ]
            if len(successes) >= window:
                sm, sm_idx = smooth(successes, window)
                sm_steps = [steps[i] for i in sm_idx]
                axes[ax_idx].plot(
                    sm_steps, np.array(sm) * 100,
                    color=color, linewidth=2.2, label=label,
                )
            else:
                axes[ax_idx].plot(
                    steps, [s * 100 for s in successes],
                    color=color, linewidth=1.5, label=label,
                )
            ax_idx += 1

        # ── Position Error ──
        if has_pos_err:
            pe = [(e["step"], e["pos_err"]) for e in episodes if e["pos_err"] is not None]
            if pe:
                pe_steps, pe_vals = zip(*pe)
                axes[ax_idx].plot(
                    pe_steps, pe_vals, color=color, alpha=0.15, linewidth=0.7,
                )
                if len(pe_vals) >= window:
                    sm, sm_idx = smooth(list(pe_vals), window)
                    sm_steps = [pe_steps[i] for i in sm_idx]
                    axes[ax_idx].plot(
                        sm_steps, sm, color=color, linewidth=2.2, label=label,
                    )
                else:
                    axes[ax_idx].plot(
                        pe_steps, pe_vals, color=color, linewidth=1.5, label=label,
                    )
            ax_idx += 1

    # ── Labels & formatting ──
    ax_idx = 0
    axes[ax_idx].set_ylabel("Episode Reward")
    axes[ax_idx].set_title("Training Comparison")
    axes[ax_idx].legend(loc="upper left")
    axes[ax_idx].grid(True, alpha=0.3)
    ax_idx += 1

    axes[ax_idx].set_ylabel("Episode Length")
    axes[ax_idx].legend(loc="upper left")
    axes[ax_idx].grid(True, alpha=0.3)
    ax_idx += 1

    if has_success:
        axes[ax_idx].set_ylabel("Success Rate (%)")
        axes[ax_idx].set_ylim(-5, 105)
        axes[ax_idx].legend(loc="upper left")
        axes[ax_idx].grid(True, alpha=0.3)
        ax_idx += 1

    if has_pos_err:
        axes[ax_idx].set_ylabel("Position Error (cm)")
        axes[ax_idx].legend(loc="upper left")
        axes[ax_idx].grid(True, alpha=0.3)
        ax_idx += 1

    axes[-1].set_xlabel("Training Step")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved comparison plot to {save_path}")
    else:
        plt.show()


# ── Summary Table ────────────────────────────────────────────────

def print_summary(all_episodes: list[list[dict]], labels: list[str]):
    """Print a summary table comparing final metrics."""
    print("\n" + "=" * 70)
    print(f"{'Metric':<25}", end="")
    for label in labels:
        print(f"{label:>15}", end="")
    print()
    print("-" * 70)

    for metric, key, fmt in [
        ("Episodes", None, "d"),
        ("Final Reward (last 10)", "reward", ".2f"),
        ("Final Length (last 10)", "length", ".0f"),
        ("Success Rate (last 10)", "success", ".0%"),
        ("Pos Error cm (last 10)", "pos_err", ".1f"),
    ]:
        print(f"{metric:<25}", end="")
        for episodes in all_episodes:
            if not episodes:
                print(f"{'N/A':>15}", end="")
                continue
            if key is None:
                print(f"{len(episodes):>15d}", end="")
                continue
            vals = [e[key] for e in episodes[-10:] if e[key] is not None]
            if vals:
                avg = sum(vals) / len(vals)
                print(f"{avg:>15{fmt}}", end="")
            else:
                print(f"{'N/A':>15}", end="")
        print()
    print("=" * 70)


# ── Main ─────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Compare training curves from multiple runs"
    )
    p.add_argument(
        "--logs", nargs="+", required=True,
        help="Paths to training log files (stdout captures)",
    )
    p.add_argument(
        "--labels", nargs="+",
        help="Labels for each run (default: log filenames)",
    )
    p.add_argument("--save", help="Save plot to file instead of showing")
    p.add_argument(
        "--window", type=int, default=10,
        help="Smoothing window size (default: 10)",
    )
    args = p.parse_args()

    if args.labels is None:
        args.labels = [p.rsplit("/", 1)[-1].replace(".log", "") for p in args.logs]

    if len(args.labels) != len(args.logs):
        print("Error: number of --labels must match number of --logs", file=sys.stderr)
        sys.exit(1)

    all_episodes = []
    for path in args.logs:
        try:
            eps = parse_log(path)
            all_episodes.append(eps)
            print(f"Loaded {path}: {len(eps)} episodes")
        except FileNotFoundError:
            print(f"Warning: {path} not found, skipping", file=sys.stderr)
            all_episodes.append([])

    print_summary(all_episodes, args.labels)
    plot_comparison(all_episodes, args.labels, args.save, args.window)


if __name__ == "__main__":
    main()
