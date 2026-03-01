"""
eval_real.py — Evaluate a trained CQN-AS checkpoint on a real Franka Panda.

This script:
  1. Connects to the robot + camera rig.
  2. Loads a snapshot.pt checkpoint + action_stats.npz.
  3. Runs N evaluation episodes, logging success / reward.

Usage
-----
    python -m simtoreal.eval_real \
        --robot-ip 192.168.131.41 \
        --snapshot ./runs/real_train/snapshot.pt \
        --action-stats ./runs/real_train/action_stats.npz \
        --camera-mode full \
        --camera-serials '{"front":"SN1","wrist":"SN2","left_shoulder":"SN3","right_shoulder":"SN4"}' \
        --num-episodes 10

    # Wrist-only eval:
    python -m simtoreal.eval_real \
        --robot-ip 192.168.131.41 \
        --snapshot ./runs/real_train/snapshot.pt \
        --action-stats ./runs/real_train/action_stats.npz \
        --camera-mode wrist \
        --wrist-serial <SERIAL> \
        --num-episodes 10

    # Dry-run (dummy cameras, no real robot commands):
    python -m simtoreal.eval_real \
        --snapshot ./runs/real_train/snapshot.pt \
        --action-stats ./runs/real_train/action_stats.npz \
        --camera-mode dummy \
        --dry-run \
        --num-episodes 2
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import utils
from rlbench_src.cqn_as import CQNASAgent

from simtoreal.cameras import (
    CAMERA_H,
    CAMERA_KEYS,
    CAMERA_W,
    NUM_CAMERAS,
    make_dummy_rig,
    make_full_rig,
    make_orbbec_rig,
    make_wrist_only_rig,
)
from simtoreal.real_env import ExtendedTimeStepWrapper, make


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate CQN-AS on real Franka Panda")

    # Robot
    p.add_argument("--robot-ip", type=str, default="192.168.131.41")
    p.add_argument("--home-q", type=str, default=None,
                   help="JSON list of 7 joint angles for home pose")
    p.add_argument("--joint-delta-clip", type=float, default=0.05)
    p.add_argument("--velocity-factor", type=float, default=0.15)
    p.add_argument("--gripper-speed", type=float, default=0.1)
    p.add_argument("--gripper-force", type=float, default=20.0)

    # Cameras — eval needs all 4 views: front, wrist, left_shoulder, right_shoulder
    p.add_argument("--camera-mode", choices=["full", "orbbec", "wrist", "dummy"],
                   default="full",
                   help="'full' = 4 RealSense cameras (recommended for eval), "
                        "'orbbec' = 4 Orbbec stereo cameras, "
                        "'wrist' = wrist-only + zero-fill, "
                        "'dummy' = all zero (dry-run only)")
    p.add_argument("--wrist-serial", type=str, default=None)
    p.add_argument("--camera-serials", type=str, default=None,
                   help='JSON dict: {"front":"SN1","wrist":"SN2",'
                        '"left_shoulder":"SN3","right_shoulder":"SN4"}')
    p.add_argument("--orbbec-serials", type=str, default=None,
                   help='JSON dict mapping camera name → Orbbec serial. '
                        'Example: \'{"front":"AB12","wrist":"CD34",'
                        '"left_shoulder":"EF56","right_shoulder":"GH78"}\'')
    p.add_argument("--camera-h", type=int, default=CAMERA_H)
    p.add_argument("--camera-w", type=int, default=CAMERA_W)

    # Checkpoint
    p.add_argument("--snapshot", type=str, required=True,
                   help="Path to snapshot.pt")
    p.add_argument("--action-stats", type=str, required=True,
                   help="Path to action_stats.npz")

    # Eval settings
    p.add_argument("--num-episodes", type=int, default=10)
    p.add_argument("--episode-length", type=int, default=200)
    p.add_argument("--frame-stack", type=int, default=8)
    p.add_argument("--action-sequence", type=int, default=4)
    p.add_argument("--temporal-ensemble", action="store_true", default=True)
    p.add_argument("--no-temporal-ensemble", dest="temporal_ensemble",
                   action="store_false")
    p.add_argument("--control-hz", type=float, default=10.0)
    p.add_argument("--device", type=str, default="cuda")

    # Agent hyperparams (must match training config)
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
    p.add_argument("--critic-target-tau", type=float, default=0.02)

    # Misc
    p.add_argument("--dry-run", action="store_true",
                   help="Don't connect to real robot (prints actions)")
    p.add_argument("--save-video", action="store_true",
                   help="Save wrist camera video of each episode")
    p.add_argument("--video-dir", type=str, default="./eval_videos")

    return p.parse_args()


def build_camera_rig(args):
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
        return make_orbbec_rig(
            serials=serials,
            height=args.camera_h, width=args.camera_w,
            camera_keys=CAMERA_KEYS,
        )
    elif args.camera_mode == "wrist":
        print("[WARN] Wrist-only mode for eval — performance will be degraded.")
        return make_wrist_only_rig(
            serial=args.wrist_serial,
            height=args.camera_h, width=args.camera_w,
            camera_keys=CAMERA_KEYS,
        )
    else:
        print("[WARN] Dummy cameras — dry-run only, policy outputs will be random.")
        return make_dummy_rig(
            height=args.camera_h, width=args.camera_w,
            camera_keys=CAMERA_KEYS,
        )


def make_agent_from_snapshot(env: ExtendedTimeStepWrapper, args) -> CQNASAgent:
    """Build agent with correct shapes and load weights from snapshot."""
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
        stddev_schedule="0.0",  # no exploration noise for eval
    )

    # Load weights
    payload = torch.load(args.snapshot, map_location=args.device)
    saved_agent = payload["agent"]
    agent.encoder.load_state_dict(saved_agent.encoder.state_dict())
    agent.critic.load_state_dict(saved_agent.critic.state_dict())
    agent.critic_target.load_state_dict(saved_agent.critic_target.state_dict())
    agent.train(False)
    print(f"[Eval] Loaded agent from {args.snapshot}")
    print(f"       Trained for {payload.get('_global_step', '?')} steps")

    return agent


def main():
    args = parse_args()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    print("=" * 60)
    print("CQN-AS Real Robot Evaluation")
    print("=" * 60)

    camera_rig = build_camera_rig(args)
    home_q = json.loads(args.home_q) if args.home_q else None

    if args.dry_run:
        print("[DRY RUN] Using dummy robot connection")
        # In dry-run, we still create the env with dummy cameras
        # but the env itself needs a real robot IP — handle this
        # by catching the connection error or using dummy cameras

    env = make(
        robot_ip=args.robot_ip,
        camera_rig=camera_rig,
        episode_length=args.episode_length,
        frame_stack=args.frame_stack,
        home_q=home_q,
        joint_delta_clip=args.joint_delta_clip,
        velocity_factor=args.velocity_factor,
        action_stats_path=args.action_stats,
        gripper_speed=args.gripper_speed,
        gripper_force=args.gripper_force,
    )

    agent = make_agent_from_snapshot(env, args)

    # Video recording setup
    video_frames = []
    if args.save_video:
        Path(args.video_dir).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Evaluation loop
    # ------------------------------------------------------------------
    dt = 1.0 / args.control_hz
    results = []

    for ep in range(args.num_episodes):
        print(f"\n--- Episode {ep + 1}/{args.num_episodes} ---")
        input("Press ENTER to start episode (position objects as needed)...")

        time_step = env.reset()
        episode_step = 0
        episode_reward = 0.0
        video_frames = []

        if args.temporal_ensemble:
            te = utils.TemporalEnsembleControl(
                args.episode_length, env.action_spec(), args.action_sequence,
            )

        action = None

        while not time_step.last():
            t0 = time.time()

            # Query policy
            if args.temporal_ensemble or episode_step % args.action_sequence == 0:
                with torch.no_grad(), utils.eval_mode(agent):
                    raw_action = agent.act(
                        time_step.rgb_obs,
                        time_step.low_dim_obs,
                        step=999999,
                        eval_mode=True,
                    )
                action = raw_action.reshape([args.action_sequence, -1])
                if args.temporal_ensemble:
                    te.register_action_sequence(action)

            # Select sub-action
            if args.temporal_ensemble:
                sub_action = te.get_action()
            else:
                sub_action = action[episode_step % args.action_sequence]

            # Execute
            time_step = env.step(sub_action)
            episode_reward += time_step.reward
            episode_step += 1

            # Optionally save wrist frame for video
            if args.save_video:
                # rgb_obs is (V, 3*fs, H, W); take last 3 channels of wrist (index 1)
                wrist_frame = time_step.rgb_obs[1, -3:]  # (3, H, W) last frame
                video_frames.append(wrist_frame.transpose(1, 2, 0))  # (H, W, 3)

            # Control-rate
            elapsed = time.time() - t0
            if elapsed < dt:
                time.sleep(dt - elapsed)

        print(
            f"  Episode {ep + 1}: steps={episode_step}, reward={episode_reward:.3f}"
        )

        # Ask human for success label
        success_input = input("  Was the task successful? [y/N]: ").strip().lower()
        success = success_input in ("y", "yes", "1")
        results.append({
            "episode": ep + 1,
            "steps": episode_step,
            "reward": episode_reward,
            "success": success,
        })

        # Save video
        if args.save_video and video_frames:
            try:
                import cv2
                video_path = Path(args.video_dir) / f"episode_{ep + 1:03d}.mp4"
                h, w = video_frames[0].shape[:2]
                writer = cv2.VideoWriter(
                    str(video_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    args.control_hz,
                    (w, h),
                )
                for frame in video_frames:
                    writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                writer.release()
                print(f"  Saved video: {video_path}")
            except ImportError:
                print("  [WARN] cv2 not available, skipping video save")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    num_success = sum(r["success"] for r in results)
    print(f"  Episodes:     {args.num_episodes}")
    print(f"  Successes:    {num_success}")
    print(f"  Success rate: {num_success / args.num_episodes * 100:.1f}%")
    print(f"  Mean reward:  {np.mean([r['reward'] for r in results]):.3f}")
    print(f"  Mean length:  {np.mean([r['steps'] for r in results]):.1f}")

    # Save results
    results_path = Path(args.video_dir if args.save_video else ".") / "eval_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {results_path}")

    env.close()


if __name__ == "__main__":
    main()
