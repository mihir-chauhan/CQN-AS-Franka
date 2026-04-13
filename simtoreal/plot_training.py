"""
Parse training stdout and plot graphs.

Usage:
    # If training is still running, tee the output:
    #   python train_real.py ... 2>&1 | tee train_log.txt
    #
    # Then after (or during):
    python plot_training.py train_log.txt

    # Or paste terminal output into a file and run:
    python plot_training.py train_log.txt --save plots.png
"""

import argparse
import re
import matplotlib.pyplot as plt
import numpy as np


def parse_log(path: str):
    episodes = []
    eval_points = []

    ep_pattern = re.compile(
        r"Episode\s+(\d+):\s+reward=([\d\.\-]+),\s+length=(\d+),.*step=(\d+)"
    )
    success_pattern = re.compile(r"\[([✓✗])\s*AutoReward\]\s+pos_err=([\d\.]+)cm.*quat_err=([\d\.]+)")
    eval_pattern = re.compile(r"\[Eval\]\s+(\d+)/(\d+)\s+succeeded\s+\((\d+)%\)")
    eval_reward_pattern = re.compile(r"\[Eval\]\s+Mean reward:\s+([\d\.\-]+)")

    with open(path, "r") as f:
        lines = f.readlines()

    pending_ep = None
    for line in lines:
        line = line.strip()

        m = ep_pattern.search(line)
        if m:
            pending_ep = {
                "episode": int(m.group(1)),
                "reward": float(m.group(2)),
                "length": int(m.group(3)),
                "step": int(m.group(4)),
                "success": None,
                "pos_err": None,
                "quat_err": None,
            }
            episodes.append(pending_ep)
            continue

        m = success_pattern.search(line)
        if m and pending_ep is not None:
            pending_ep["success"] = 1 if m.group(1) == "✓" else 0
            pending_ep["pos_err"] = float(m.group(2))
            pending_ep["quat_err"] = float(m.group(3))
            continue

        # Also catch the simpler success/fail lines
        if "Success! reward=1.0" in line and pending_ep is not None:
            if pending_ep["success"] is None:
                pending_ep["success"] = 1
        elif "No success." in line and pending_ep is not None:
            if pending_ep["success"] is None:
                pending_ep["success"] = 0

        m = eval_pattern.search(line)
        if m:
            eval_points.append({
                "succeeded": int(m.group(1)),
                "total": int(m.group(2)),
                "pct": int(m.group(3)),
            })

    return episodes, eval_points


def plot(episodes, eval_points, save_path=None):
    if not episodes:
        print("No episodes found in log.")
        return

    eps = [e["episode"] for e in episodes]
    steps = [e["step"] for e in episodes]
    rewards = [e["reward"] for e in episodes]
    lengths = [e["length"] for e in episodes]

    has_success = any(e["success"] is not None for e in episodes)
    has_pos_err = any(e["pos_err"] is not None for e in episodes)

    n_plots = 2 + int(has_success) + int(has_pos_err)
    fig, axes = plt.subplots(n_plots, 1, figsize=(10, 3.5 * n_plots), sharex=True)
    if n_plots == 1:
        axes = [axes]

    idx = 0

    # --- Episode reward ---
    axes[idx].plot(steps, rewards, "b-", alpha=0.4, linewidth=0.8)
    if len(rewards) >= 5:
        window = min(10, len(rewards))
        smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
        axes[idx].plot(steps[window - 1:], smoothed, "b-", linewidth=2,
                       label=f"{window}-ep moving avg")
        axes[idx].legend()
    axes[idx].set_ylabel("Episode Reward")
    axes[idx].set_title("Training Progress")
    axes[idx].grid(True, alpha=0.3)
    idx += 1

    # --- Episode length ---
    axes[idx].plot(steps, lengths, "g-", alpha=0.5, linewidth=0.8)
    axes[idx].set_ylabel("Episode Length")
    axes[idx].grid(True, alpha=0.3)
    idx += 1

    # --- Success rate (if available) ---
    if has_success:
        successes = [e["success"] if e["success"] is not None else 0 for e in episodes]
        s_steps = steps
        if len(successes) >= 5:
            window = min(10, len(successes))
            sr = np.convolve(successes, np.ones(window) / window, mode="valid")
            axes[idx].plot(s_steps[window - 1:], sr * 100, "r-", linewidth=2)
            axes[idx].set_ylabel("Success Rate (%)")
        else:
            axes[idx].bar(s_steps, [s * 100 for s in successes], width=50, color="r", alpha=0.6)
            axes[idx].set_ylabel("Success (%)")
        axes[idx].set_ylim(-5, 105)
        axes[idx].grid(True, alpha=0.3)
        idx += 1

    # --- Position error (if available) ---
    if has_pos_err:
        pe = [(e["step"], e["pos_err"]) for e in episodes if e["pos_err"] is not None]
        if pe:
            pe_steps, pe_vals = zip(*pe)
            axes[idx].plot(pe_steps, pe_vals, "m-", alpha=0.5, linewidth=0.8)
            if len(pe_vals) >= 5:
                window = min(10, len(pe_vals))
                smoothed = np.convolve(pe_vals, np.ones(window) / window, mode="valid")
                axes[idx].plot(pe_steps[window - 1:], smoothed, "m-", linewidth=2)
            axes[idx].set_ylabel("Position Error (cm)")
            axes[idx].grid(True, alpha=0.3)
        idx += 1

    axes[-1].set_xlabel("Training Step")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {save_path}")
    else:
        plt.show()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("log", help="Path to training log (stdout capture)")
    p.add_argument("--save", default=None, help="Save plot to file instead of showing")
    args = p.parse_args()

    episodes, eval_points = parse_log(args.log)
    print(f"Parsed {len(episodes)} episodes, {len(eval_points)} eval checkpoints")

    if eval_points:
        for ep in eval_points:
            print(f"  Eval: {ep['succeeded']}/{ep['total']} ({ep['pct']}%)")

    plot(episodes, eval_points, save_path=args.save)


if __name__ == "__main__":
    main()
