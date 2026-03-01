"""
train_real.py — Train CQN-AS on a real Franka Panda.

Workflow
--------
1. Connect to robot + cameras (wrist-only is fine for training).
2. Record kinesthetic demonstrations (gravity-comp / freedrive mode).
3. Compute action stats from demos.
4. Train CQN-AS with the same agent architecture as RLBench training.
5. Periodically save snapshots + action stats.

Usage
-----
    python -m simtoreal.train_real \
        --robot-ip 192.168.131.41 \
        --wrist-serial <REALSENSE_SERIAL> \
        --num-demos 10 \
        --num-train-steps 50000 \
        --save-dir ./runs/real_train

    # Resume from checkpoint:
    python -m simtoreal.train_real \
        --robot-ip 192.168.131.41 \
        --wrist-serial <SERIAL> \
        --resume ./runs/real_train/snapshot.pt \
        --num-train-steps 100000

Camera modes
------------
    --camera-mode wrist       Only wrist RealSense (other views zero-filled)
    --camera-mode full        All 4 cameras via RealSense (provide --camera-serials)
    --camera-mode dummy       All dummy cameras (for testing without hardware)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from dm_env import specs

# Add project root to path so imports work
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import utils
from logger import Logger
from rlbench_src.cqn_as import CQNASAgent
from rlbench_src.replay_buffer_action_sequence import (
    ReplayBufferStorage,
    make_replay_loader,
)

from simtoreal.cameras import (
    CAMERA_H,
    CAMERA_KEYS,
    CAMERA_W,
    NUM_CAMERAS,
    CameraRig,
    make_dummy_rig,
    make_full_rig,
    make_orbbec_rig,
    make_wrist_only_rig,
)
from simtoreal.real_env import ExtendedTimeStepWrapper, RealFrankaEnv, make


# ============================================================================
# Argument parsing
# ============================================================================


def parse_args():
    p = argparse.ArgumentParser(description="Train CQN-AS on real Franka Panda")

    # Robot
    p.add_argument("--robot-ip", type=str, default="192.168.131.41")
    p.add_argument("--home-q", type=str, default=None,
                   help="JSON list of 7 joint angles for home pose")
    p.add_argument("--joint-delta-clip", type=float, default=0.05)
    p.add_argument("--velocity-factor", type=float, default=0.15)

    # Cameras — training requires all 4 views: front, wrist, left_shoulder, right_shoulder
    p.add_argument("--camera-mode", choices=["full", "orbbec", "wrist", "dummy"],
                   default="full",
                   help="'full' = 4 RealSense cameras, "
                        "'orbbec' = 4 Orbbec stereo cameras, "
                        "'wrist' = wrist-only + zero-fill other 3 views, "
                        "'dummy' = all zero (code testing only)")
    p.add_argument("--wrist-serial", type=str, default=None,
                   help="RealSense serial for wrist camera (used in wrist mode)")
    p.add_argument("--camera-serials", type=str, default=None,
                   help='JSON dict mapping camera name → RealSense serial. '
                        'Keys: front, wrist, left_shoulder, right_shoulder. '
                        'Example: \'{"front":"SN1","wrist":"SN2",'
                        '"left_shoulder":"SN3","right_shoulder":"SN4"}\'')
    p.add_argument("--orbbec-serials", type=str, default=None,
                   help='JSON dict mapping camera name → Orbbec serial. '
                        'Keys: front, wrist, left_shoulder, right_shoulder. '
                        'Example: \'{"front":"AB12","wrist":"CD34",'
                        '"left_shoulder":"EF56","right_shoulder":"GH78"}\'')
    p.add_argument("--camera-h", type=int, default=CAMERA_H)
    p.add_argument("--camera-w", type=int, default=CAMERA_W)

    # Demos
    p.add_argument("--num-demos", type=int, default=10)
    p.add_argument("--demo-hz", type=float, default=10.0)
    p.add_argument("--demo-max-steps", type=int, default=500)
    p.add_argument("--demo-dir", type=str, default=None,
                   help="If set, save/load demos to/from this directory")
    p.add_argument("--load-demos-only", action="store_true",
                   help="Load demos from --demo-dir instead of recording")

    # Training
    p.add_argument("--num-train-steps", type=int, default=50000)
    p.add_argument("--episode-length", type=int, default=200)
    p.add_argument("--frame-stack", type=int, default=8)
    p.add_argument("--action-sequence", type=int, default=4)
    p.add_argument("--temporal-ensemble", action="store_true", default=True)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--demo-batch-size", type=int, default=256)
    p.add_argument("--replay-buffer-size", type=int, default=1000000)
    p.add_argument("--num-update-steps", type=int, default=1)
    p.add_argument("--eval-every", type=int, default=2500)
    p.add_argument("--num-eval-episodes", type=int, default=5)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", type=str, default="cuda")

    # Agent hyperparams (match config_cqn_as_rlbench.yaml defaults)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--feature-dim", type=int, default=64)
    p.add_argument("--hidden-dim", type=int, default=512)
    p.add_argument("--levels", type=int, default=3)
    p.add_argument("--bins", type=int, default=5)
    p.add_argument("--atoms", type=int, default=51)
    p.add_argument("--v-min", type=float, default=-2.0)
    p.add_argument("--v-max", type=float, default=2.0)
    p.add_argument("--bc-lambda", type=float, default=1.0)
    p.add_argument("--bc-margin", type=float, default=0.01)
    p.add_argument("--critic-lambda", type=float, default=0.1)
    p.add_argument("--stddev-schedule", type=str, default="0.01")
    p.add_argument("--critic-target-tau", type=float, default=0.02)

    # Save / resume
    p.add_argument("--save-dir", type=str, default="./runs/real_train")
    p.add_argument("--resume", type=str, default=None,
                   help="Path to snapshot.pt to resume from")
    p.add_argument("--save-every", type=int, default=5000)

    # Control
    p.add_argument("--control-hz", type=float, default=10.0)

    return p.parse_args()


# ============================================================================
# Camera setup
# ============================================================================


def build_camera_rig(args) -> CameraRig:
    """Build camera rig matching training config: 4 cameras at 84×84 RGB.

    Camera positions to replicate from RLBench:
      - front:           ~1m in front of robot, chest height, facing robot
      - wrist:           mounted on gripper (Intel RealSense recommended)
      - left_shoulder:   ~0.5m above left shoulder, angled down at workspace
      - right_shoulder:  ~0.5m above right shoulder, angled down at workspace
    """
    if args.camera_mode == "full":
        serials = json.loads(args.camera_serials) if args.camera_serials else {}
        missing = [k for k in CAMERA_KEYS if k not in serials]
        if missing:
            print(f"[WARN] No serial provided for cameras: {missing}")
            print(f"       These will use DummyCamera (zero frames).")
            print(f"       For best results, provide all 4 serials.")
        return make_full_rig(
            serials=serials,
            height=args.camera_h, width=args.camera_w,
            camera_keys=CAMERA_KEYS,
        )
    elif args.camera_mode == "orbbec":
        serials = json.loads(args.orbbec_serials) if args.orbbec_serials else {}
        missing = [k for k in CAMERA_KEYS if k not in serials]
        if missing:
            print(f"[WARN] No Orbbec serial provided for cameras: {missing}")
            print(f"       These will use DummyCamera (zero frames).")
            print(f"       For best results, provide all 4 serials.")
        return make_orbbec_rig(
            serials=serials,
            height=args.camera_h, width=args.camera_w,
            camera_keys=CAMERA_KEYS,
        )
    elif args.camera_mode == "wrist":
        print("[WARN] Wrist-only mode: front, left_shoulder, right_shoulder "
              "will be zero-filled. Policy quality will be degraded.")
        return make_wrist_only_rig(
            serial=args.wrist_serial,
            height=args.camera_h, width=args.camera_w,
            camera_keys=CAMERA_KEYS,
        )
    else:
        print("[WARN] Dummy camera mode: ALL views are zero-filled. "
              "Only use for code testing!")
        return make_dummy_rig(
            height=args.camera_h, width=args.camera_w,
            camera_keys=CAMERA_KEYS,
        )


# ============================================================================
# Demo collection / loading
# ============================================================================


def collect_demos(env: RealFrankaEnv, args) -> list[list]:
    """Record kinesthetic demonstrations interactively."""
    demos = []
    for i in range(args.num_demos):
        input(
            f"\n[Demo {i+1}/{args.num_demos}] "
            "Put robot in freedrive mode, position it at the start, "
            "then press ENTER to begin recording..."
        )
        demo = env.record_demo(
            hz=args.demo_hz,
            max_steps=args.demo_max_steps,
        )
        demos.append(demo)

        # Optionally save each demo immediately
        if args.demo_dir:
            demo_path = Path(args.demo_dir)
            demo_path.mkdir(parents=True, exist_ok=True)
            save_demo(demo, demo_path / f"demo_{i:04d}.npz")

    return demos


def save_demo(demo, path: Path):
    """Save a single demo to disk."""
    data = {
        "rgb_obs": np.stack([ts.rgb_obs for ts in demo]),
        "low_dim_obs": np.stack([ts.low_dim_obs for ts in demo]),
        "action": np.stack([ts.action for ts in demo]),
        "reward": np.array([ts.reward for ts in demo]),
        "discount": np.array([ts.discount for ts in demo]),
        "step_type": np.array([ts.step_type for ts in demo]),
    }
    np.savez_compressed(str(path), **data)
    print(f"  Saved demo to {path}")


def load_demos(demo_dir: str, env: RealFrankaEnv) -> list[list]:
    """Load demos from a directory of .npz files."""
    from dm_env import StepType
    from simtoreal.real_env import ExtendedTimeStep

    demo_dir = Path(demo_dir)
    demo_files = sorted(demo_dir.glob("demo_*.npz"))
    if not demo_files:
        raise FileNotFoundError(f"No demo files found in {demo_dir}")

    demos = []
    for path in demo_files:
        data = np.load(str(path))
        demo = []
        for i in range(len(data["action"])):
            demo.append(
                ExtendedTimeStep(
                    rgb_obs=data["rgb_obs"][i],
                    low_dim_obs=data["low_dim_obs"][i],
                    action=data["action"][i],
                    reward=float(data["reward"][i]),
                    discount=float(data["discount"][i]),
                    step_type=StepType(int(data["step_type"][i])),
                    demo=1.0,
                )
            )
        demos.append(demo)
        print(f"  Loaded {path.name} ({len(demo)} steps)")
    return demos


# ============================================================================
# Agent construction
# ============================================================================


def make_agent(env: ExtendedTimeStepWrapper, args) -> CQNASAgent:
    rgb_spec = env.rgb_observation_spec()
    low_dim_spec = env.low_dim_observation_spec()
    action_spec = env.action_spec()

    action_shape = (args.action_sequence, *action_spec.shape)

    agent = CQNASAgent(
        rgb_obs_shape=rgb_spec.shape,
        low_dim_obs_shape=low_dim_spec.shape,
        action_shape=action_shape,
        device=args.device,
        lr=args.lr,
        feature_dim=args.feature_dim,
        hidden_dim=args.hidden_dim,
        levels=args.levels,
        bins=args.bins,
        atoms=args.atoms,
        v_min=args.v_min,
        v_max=args.v_max,
        bc_lambda=args.bc_lambda,
        bc_margin=args.bc_margin,
        gru_layers=1,
        rgb_encoder_layers=0,
        use_parallel_impl=False,
        critic_lambda=args.critic_lambda,
        critic_target_tau=args.critic_target_tau,
        critic_target_interval=1,
        weight_decay=args.weight_decay,
        num_expl_steps=0,
        update_every_steps=1,
        stddev_schedule=args.stddev_schedule,
    )
    return agent


# ============================================================================
# Main
# ============================================================================


def main():
    args = parse_args()
    utils.set_seed_everywhere(args.seed)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Setup cameras + environment
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Setting up cameras...")
    camera_rig = build_camera_rig(args)

    home_q = json.loads(args.home_q) if args.home_q else None

    print("Connecting to robot...")
    action_stats_path = str(save_dir / "action_stats.npz") \
        if (save_dir / "action_stats.npz").exists() else None

    env = make(
        robot_ip=args.robot_ip,
        camera_rig=camera_rig,
        episode_length=args.episode_length,
        frame_stack=args.frame_stack,
        home_q=home_q,
        joint_delta_clip=args.joint_delta_clip,
        velocity_factor=args.velocity_factor,
        action_stats_path=action_stats_path,
    )

    # ------------------------------------------------------------------
    # 2. Collect or load demonstrations
    # ------------------------------------------------------------------
    print("=" * 60)
    real_env: RealFrankaEnv = env._env  # unwrap to access demo recording

    if args.load_demos_only and args.demo_dir:
        print(f"Loading demos from {args.demo_dir}...")
        demos = load_demos(args.demo_dir, real_env)
    else:
        print(f"Recording {args.num_demos} kinesthetic demos...")
        demos = collect_demos(real_env, args)

    # Compute & save action stats from demos
    action_stats = real_env.extract_action_stats(demos)
    real_env.set_action_stats(action_stats)
    real_env.save_action_stats(str(save_dir / "action_stats.npz"))

    # Rescale demo actions to [-1, 1]
    demos = [real_env.rescale_demo_actions(d) for d in demos]
    print(f"Action stats: min={action_stats['min']}, max={action_stats['max']}")

    # ------------------------------------------------------------------
    # 3. Build agent
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Building CQN-AS agent...")
    agent = make_agent(env, args)

    global_step = 0
    global_episode = 0

    if args.resume:
        print(f"Resuming from {args.resume}...")
        payload = torch.load(args.resume, map_location=args.device)
        saved_agent = payload["agent"]
        agent.encoder.load_state_dict(saved_agent.encoder.state_dict())
        agent.critic.load_state_dict(saved_agent.critic.state_dict())
        agent.critic_target.load_state_dict(saved_agent.critic_target.state_dict())
        global_step = payload.get("_global_step", 0)
        global_episode = payload.get("_global_episode", 0)
        print(f"  Resumed at step {global_step}, episode {global_episode}")

    # ------------------------------------------------------------------
    # 4. Setup replay buffers
    # ------------------------------------------------------------------
    data_specs = (
        env.rgb_raw_observation_spec(),
        env.low_dim_raw_observation_spec(),
        env.action_spec(),
        specs.Array((1,), np.float32, "reward"),
        specs.Array((1,), np.float32, "discount"),
        specs.Array((1,), np.float32, "demo"),
    )
    replay_storage = ReplayBufferStorage(
        data_specs, save_dir / "buffer", use_relabeling=True,
    )
    demo_replay_storage = ReplayBufferStorage(
        data_specs, save_dir / "demo_buffer", use_relabeling=True,
    )

    # Insert demos into replay buffers
    for demo in demos:
        for ts in demo:
            replay_storage.add(ts)
            demo_replay_storage.add(ts)
    print(f"Loaded {len(replay_storage)} demo transitions into buffers")

    replay_loader = make_replay_loader(
        replay_storage,
        args.replay_buffer_size,
        args.batch_size,
        4,  # num_workers
        save_snapshot=False,
        nstep=1,
        discount=0.99,
        frame_stack=args.frame_stack,
        fill_action="zero_action",
    )
    demo_replay_loader = make_replay_loader(
        demo_replay_storage,
        args.replay_buffer_size,
        args.demo_batch_size,
        4,
        save_snapshot=False,
        nstep=1,
        discount=0.99,
        frame_stack=args.frame_stack,
        fill_action="zero_action",
    )
    replay_iter = None

    def get_replay_iter():
        nonlocal replay_iter
        if replay_iter is None:
            replay_iter = utils.DemoMergedIterator(
                iter(replay_loader), iter(demo_replay_loader)
            )
        return replay_iter

    # Temporal ensemble
    if args.temporal_ensemble:
        temporal_ensemble = utils.TemporalEnsembleControl(
            args.episode_length,
            env.action_spec(),
            args.action_sequence,
        )

    # ------------------------------------------------------------------
    # 5. Training loop
    # ------------------------------------------------------------------
    print("=" * 60)
    print(f"Starting training for {args.num_train_steps} steps...")

    dt = 1.0 / args.control_hz
    episode_step = 0
    episode_reward = 0.0

    time_step = env.reset()
    if args.temporal_ensemble:
        temporal_ensemble.reset()
    replay_storage.add(time_step)
    demo_replay_storage.add(time_step)

    timer = utils.Timer()

    while global_step < args.num_train_steps:
        # Episode boundary
        if time_step.last():
            global_episode += 1
            elapsed_time, total_time = timer.reset()
            episode_frame = episode_step
            print(
                f"  Episode {global_episode}: "
                f"reward={episode_reward:.2f}, "
                f"length={episode_step}, "
                f"fps={episode_frame / max(elapsed_time, 1e-6):.1f}, "
                f"step={global_step}"
            )

            # Eval
            if global_step >= args.eval_every and global_step % args.eval_every < episode_step:
                print("  [Eval] Running evaluation episodes...")
                eval_reward = run_eval(
                    env, agent, args.num_eval_episodes,
                    args.action_sequence, args.temporal_ensemble,
                    args.episode_length,
                )
                print(f"  [Eval] Mean reward: {eval_reward:.3f}")

            # Save
            if global_step % args.save_every < episode_step:
                save_snapshot(save_dir, agent, global_step, global_episode)

            time_step = env.reset()
            if args.temporal_ensemble:
                temporal_ensemble.reset()
            replay_storage.add(time_step)
            demo_replay_storage.add(time_step)
            episode_step = 0
            episode_reward = 0.0

        # Sample action
        t0 = time.time()
        if args.temporal_ensemble or episode_step % args.action_sequence == 0:
            with torch.no_grad(), utils.eval_mode(agent):
                action = agent.act(
                    time_step.rgb_obs,
                    time_step.low_dim_obs,
                    global_step,
                    eval_mode=False,
                )
            action = action.reshape([args.action_sequence, -1])
            if args.temporal_ensemble:
                temporal_ensemble.register_action_sequence(action)

        # Update agent
        if global_step > 0 and global_step % 1 == 0:
            for _ in range(args.num_update_steps):
                batch = next(get_replay_iter())
                batch = utils.to_torch_pixel_tensor_dict(batch, args.device)
                metrics = agent.update(batch)
                agent.update_target_critic(global_step)

        # Execute action
        if args.temporal_ensemble:
            sub_action = temporal_ensemble.get_action()
        else:
            sub_action = action[episode_step % args.action_sequence]
        sub_action = agent.add_noise_to_action(sub_action, global_step)

        time_step = env.step(sub_action)
        episode_reward += time_step.reward
        replay_storage.add(time_step)
        demo_replay_storage.add(time_step)
        episode_step += 1
        global_step += 1

        # Control-rate sleep
        elapsed = time.time() - t0
        if elapsed < dt:
            time.sleep(dt - elapsed)

    # Final save
    save_snapshot(save_dir, agent, global_step, global_episode)
    print(f"\nTraining complete. {global_step} steps, {global_episode} episodes.")
    env.close()


# ============================================================================
# Eval helper
# ============================================================================


def run_eval(
    env: ExtendedTimeStepWrapper,
    agent: CQNASAgent,
    num_episodes: int,
    action_sequence: int,
    temporal_ensemble: bool,
    episode_length: int,
) -> float:
    total_reward = 0.0
    for ep in range(num_episodes):
        episode_step = 0
        time_step = env.reset()
        if temporal_ensemble:
            te = utils.TemporalEnsembleControl(
                episode_length, env.action_spec(), action_sequence,
            )
        while not time_step.last():
            if temporal_ensemble or episode_step % action_sequence == 0:
                with torch.no_grad(), utils.eval_mode(agent):
                    action = agent.act(
                        time_step.rgb_obs,
                        time_step.low_dim_obs,
                        step=999999,
                        eval_mode=True,
                    )
                action = action.reshape([action_sequence, -1])
                if temporal_ensemble:
                    te.register_action_sequence(action)
            if temporal_ensemble:
                sub_action = te.get_action()
            else:
                sub_action = action[episode_step % action_sequence]
            time_step = env.step(sub_action)
            total_reward += time_step.reward
            episode_step += 1
    return total_reward / max(num_episodes, 1)


# ============================================================================
# Snapshot
# ============================================================================


def save_snapshot(save_dir: Path, agent, global_step, global_episode):
    path = save_dir / "snapshot.pt"
    payload = {
        "agent": agent,
        "timer": None,
        "_global_step": global_step,
        "_global_episode": global_episode,
    }
    torch.save(payload, str(path))
    print(f"  Saved snapshot to {path} (step={global_step})")


if __name__ == "__main__":
    main()
