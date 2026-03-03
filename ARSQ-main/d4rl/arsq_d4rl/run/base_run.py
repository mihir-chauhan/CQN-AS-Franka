import os
from pathlib import Path

import numpy as np
import skvideo.io
import wandb


def print_metrics(metrics, name, epoch, total_time, exp_name=""):
    print(f"\n{name}: {epoch}, time: {total_time:.4f}")
    if exp_name:
        print(f"exp_name: {exp_name}")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

def print_metrics_str(metrics, name, epoch, total_time, exp_name=""):
    output = f"\n{name}: {epoch}, time: {total_time:.4f}\n"
    if exp_name:
        output += f"exp_name: {exp_name}\n"
    for k, v in metrics.items():
        output += f"{k}: {v:.4f}\n"

    return output


def prefix_metrics(prefix, d):
    return {f"{prefix}/{k}": v for k, v in d.items()}


def get_time_metrics(rollout_timer, train_timer, eval_timer):
    # time
    time_metrics = {}
    epoch_time = 0
    if rollout_timer is not None:
        time_metrics['rollout_time'] = rollout_timer()
        epoch_time += rollout_timer()
    if train_timer is not None:
        time_metrics['train_time'] = train_timer()
        epoch_time += train_timer()
    if eval_timer is not None:
        time_metrics['eval_time'] = eval_timer()
        epoch_time += eval_timer()
    time_metrics['epoch_time'] = epoch_time

    return time_metrics


def log(epoch, total_steps, expl_metrics, train_metrics, eval_metrics, time_metrics):
    # log
    metrics = {}
    metrics["epoch"] = epoch
    metrics['env_steps'] = total_steps

    if expl_metrics is not None:
        metrics.update(prefix_metrics("expl", expl_metrics))
    metrics.update(train_metrics)  # train metrics with its own prefix
    if eval_metrics is not None:
        metrics.update(prefix_metrics("eval", eval_metrics))
    metrics.update(prefix_metrics("time", time_metrics))

    wandb.log(metrics)


def stat_traj(trajs, env, render_path: Path = None):
    metrics = {}

    metrics['average_return'] = np.mean([np.sum(t['rewards']) for t in trajs])
    metrics['average_traj_length'] = np.mean([len(t['rewards']) for t in trajs])
    if 'goal_achieved' in trajs[0]:
        # for adroit envs
        metrics['goal_achieved_rate'] = np.mean([1 in t['goal_achieved'] for t in trajs])
        metrics['average_normalized_return'] = metrics['goal_achieved_rate']
    else:
        # for d4rl envs
        metrics['average_normalized_return'] = np.mean(
            [env.get_normalized_score(np.sum(t['rewards'])) for t in trajs])

    # render
    if (render_path is None) and (trajs[0].get("imgs") is not None):
        print("Rendered images is not saved!")
    if render_path is not None:
        os.makedirs(render_path, exist_ok=True)
        for idx, traj in enumerate(trajs):
            if traj.get("imgs") is not None:
                imgs = traj["imgs"]
                if 'goal_achieved' in traj:
                    reward = np.sum(traj['rewards'])
                else:
                    reward = env.get_normalized_score(np.sum(traj['rewards']))
                skvideo.io.vwrite(str(render_path / f"traj{idx}_{reward:.2f}.mp4"), imgs)

    return metrics
