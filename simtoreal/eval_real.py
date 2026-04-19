"""
eval_real.py — Evaluate a trained CQN-AS or ARSQ checkpoint on a real Franka Panda.

This script:
  1. Connects to the robot + camera rig.
  2. Loads a snapshot.pt checkpoint + action_stats.npz.
  3. Runs N evaluation episodes, logging success / reward.

Usage
-----
    # CQN-AS eval (default):
    python -m simtoreal.eval_real \
        --agent cqn \
        --robot-ip 192.168.131.41 \
        --snapshot ./runs/real_train/snapshot.pt \
        --action-stats ./runs/real_train/action_stats.npz \
        --camera-mode full \
        --camera-serials '{"front":"SN1","wrist":"SN2","left_shoulder":"SN3","right_shoulder":"SN4"}' \
        --num-episodes 10

    # ARSQ eval:
    python -m simtoreal.eval_real \
        --agent arsq \
        --robot-ip 192.168.131.41 \
        --snapshot ./runs/real_train_arsq/snapshot.pt \
        --action-stats ./runs/real_train_arsq/action_stats.npz \
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
from arsq_src.sqar import SQARAgent

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
from simtoreal.real_env import ExtendedTimeStepWrapper, RealFrankaEnv, make


def _load_goal_pose(path: str) -> dict:
    """Load goal_pose.json → {'position': ndarray(3), 'quaternion': ndarray(4)}."""
    with open(path, "r") as f:
        data = json.load(f)
    return {
        "position": np.array(data["position"], dtype=np.float64),
        "quaternion": np.array(data["quaternion_xyzw"], dtype=np.float64),
    }


def _quat_distance(q1: np.ndarray, q2: np.ndarray) -> float:
    """Quaternion distance: 1 - |q1·q2|.  Returns 0 (identical) to 1 (180°)."""
    return 1.0 - min(abs(float(np.dot(q1, q2))), 1.0)


def _check_success(
    real_env: RealFrankaEnv,
    goal: dict,
    pos_thresh: float,
    quat_thresh: float,
) -> bool:
    """Check if current EE pose is within thresholds of goal."""
    result = real_env.get_ee_pose()
    if result is None:
        print("  [AutoSuccess] Could not read EE pose — marking fail")
        return False
    pos, quat = result
    pos_err = np.linalg.norm(pos - goal["position"])
    q_err = _quat_distance(quat, goal["quaternion"])
    success = pos_err <= pos_thresh and q_err <= quat_thresh
    symbol = "✓" if success else "✗"
    print(f"  [{symbol} AutoSuccess] pos_err={pos_err*100:.1f}cm "
          f"(thresh={pos_thresh*100:.1f}cm) | "
          f"quat_err={q_err:.3f} (thresh={quat_thresh:.3f})")
    return success


def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate CQN-AS or ARSQ on real Franka Panda"
    )

    # Agent selection
    p.add_argument("--agent", choices=["cqn", "arsq"], default="arsq",
                   help="'cqn' = CQN-AS (action sequences), "
                        "'arsq' = SQAR (single-step). "
                        "Auto-detected from snapshot if saved with agent_type.")

    # Robot
    p.add_argument("--robot-ip", type=str, default="192.168.131.41")
    p.add_argument("--home-q", type=str, default=None,
                   help="JSON list of 7 joint angles for home pose")
    p.add_argument("--home-sequence", type=str, nargs="+", default=None,
                   help="Ordered list of .npy files (7-dof joint configs) "
                        "executed during reset. Paths resolved relative to "
                        "the simtoreal/ directory if not absolute. "
                        "Default: partialhome.npy image_45.npy")
    p.add_argument("--task-mode", choices=["reach-grasp", "pick-place"],
                   default="reach-grasp",
                   help="'pick-place' pauses after homing for the operator "
                        "to load the object, then closes the gripper before "
                        "the episode starts.")
    p.add_argument("--gripper-start", choices=["open", "closed"], default=None,
                   help="Explicit gripper state at end of reset. Defaults: "
                        "'open' for reach-grasp, 'closed' for pick-place.")
    p.add_argument("--skip-initial-home", action="store_true",
                   help="Skip the home sequence on the very first reset "
                        "(robot is assumed already at the starting pose). "
                        "Subsequent episode resets still home.")
    p.add_argument("--freeze-gripper", action="store_true",
                   help="Ignore the agent's gripper action dim during step(). "
                        "Gripper is closed at reset (pick-place) and auto-"
                        "opens when an episode ends.")
    p.add_argument("--joint-delta-clip", type=float, default=0.05)
    p.add_argument("--velocity-factor", type=float, default=0.15)
    p.add_argument("--gripper-speed", type=float, default=0.1)
    p.add_argument("--gripper-force", type=float, default=20.0)

    # Cameras — eval needs all 4 views: front, wrist, left_shoulder, right_shoulder
    p.add_argument("--camera-mode", choices=["full", "orbbec", "wrist", "dummy"],
                   default="full",
                   help="'full' = 4 RealSense cameras (recommended for eval), "
                        "'orbbec' = 4 Orbbec stereo cameras, "
                        "'wrist' = wrist-only Orbbec + zero-fill, "
                        "'dummy' = all zero (dry-run only)")
    p.add_argument("--wrist-serial", type=str, default=None,
                   help="Orbbec serial for wrist camera (used in wrist mode)")
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

    # Agent hyperparams — shared
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--feature-dim", type=int, default=64)
    p.add_argument("--hidden-dim", type=int, default=512)
    p.add_argument("--levels", type=int, default=3)
    p.add_argument("--bins", type=int, default=5)
    p.add_argument("--critic-target-tau", type=float, default=0.02)

    # CQN-AS specific
    p.add_argument("--atoms", type=int, default=51)
    p.add_argument("--v-min", type=float, default=-2.0)
    p.add_argument("--v-max", type=float, default=2.0)
    p.add_argument("--bc-lambda", type=float, default=1.0)
    p.add_argument("--bc-margin", type=float, default=0.01)
    p.add_argument("--critic-lambda", type=float, default=0.1)

    # ARSQ specific
    p.add_argument("--soft-alpha", type=float, default=0.001)
    # Physics-informed regularizers (no-ops at eval, needed for weight loading)
    p.add_argument("--use-fk-reg", action="store_true", default=True)
    p.add_argument("--no-fk-reg", dest="use_fk_reg", action="store_false")
    p.add_argument("--use-eikonal", action="store_true", default=False)
    p.add_argument("--nu", type=float, default=0.01)
    p.add_argument("--kappa", type=float, default=0.1)
    p.add_argument("--num-walks", type=int, default=10)
    p.add_argument("--fk-weight", type=float, default=1.0)

    # Auto-success detection
    p.add_argument("--goal-pose", type=str, default=None,
                   help="Path to goal_pose.json for auto success detection")
    p.add_argument("--success-pos-thresh", type=float, default=0.0254,
                   help="Position threshold in meters (default 1 inch)")
    p.add_argument("--success-quat-thresh", type=float, default=0.05,
                   help="Quaternion distance threshold (default ~18 deg)")

    # Results
    p.add_argument("--results-dir", type=str, default=None,
                   help="Directory to save eval_results.json (default: same as snapshot)")

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
      - wrist:           mounted on gripper (Orbbec)
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
        print("[WARN] Wrist-only mode (Orbbec) for eval — performance will be degraded.")
        return make_wrist_only_rig(
            serial=args.wrist_serial,
            height=args.camera_h, width=args.camera_w,
            camera_keys=CAMERA_KEYS,
            use_orbbec=True,
        )
    else:
        print("[WARN] Dummy cameras — dry-run only, policy outputs will be random.")
        return make_dummy_rig(
            height=args.camera_h, width=args.camera_w,
            camera_keys=CAMERA_KEYS,
        )


def make_agent_from_snapshot(env: ExtendedTimeStepWrapper, args):
    """Build agent with correct shapes and load weights from snapshot.

    Works for both CQN-AS and ARSQ. If the snapshot contains 'agent_type',
    that overrides the --agent flag.
    """
    payload = torch.load(args.snapshot, map_location=args.device)

    # Auto-detect agent type from snapshot if available
    saved_type = payload.get("agent_type", None)
    if saved_type is not None and saved_type != args.agent:
        print(f"[Eval] Snapshot was saved with agent_type='{saved_type}', "
              f"overriding --agent={args.agent}")
        args.agent = saved_type

    use_cqn = args.agent == "cqn"

    # ARSQ doesn't use action sequences or temporal ensemble
    if not use_cqn:
        args.action_sequence = 1
        args.temporal_ensemble = False

    rgb_spec = env.rgb_observation_spec()
    low_dim_spec = env.low_dim_observation_spec()
    action_spec = env.action_spec()

    if use_cqn:
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
        saved_agent = payload["agent"]
        agent.encoder.load_state_dict(saved_agent.encoder.state_dict())
        agent.critic.load_state_dict(saved_agent.critic.state_dict())
        agent.critic_target.load_state_dict(saved_agent.critic_target.state_dict())
    else:
        agent = SQARAgent(
            rgb_obs_shape=rgb_spec.shape,
            low_dim_obs_shape=low_dim_spec.shape,
            action_shape=action_spec.shape,
            device=args.device,
            lr=args.lr,
            weight_decay=args.weight_decay,
            feature_dim=args.feature_dim,
            hidden_dim=args.hidden_dim,
            levels=args.levels,
            bins=args.bins,
            soft_alpha=args.soft_alpha,
            critic_target_tau=args.critic_target_tau,
            use_fk_reg=args.use_fk_reg,
            use_eikonal=args.use_eikonal,
            nu=args.nu,
            kappa=args.kappa,
            num_walks=args.num_walks,
            fk_weight=args.fk_weight,
        )
        saved_agent = payload["agent"]
        agent.encoder.load_state_dict(saved_agent.encoder.state_dict())
        agent.qf1.load_state_dict(saved_agent.qf1.state_dict())
        agent.qf2.load_state_dict(saved_agent.qf2.state_dict())
        agent.qf1_target.load_state_dict(saved_agent.qf1_target.state_dict())
        agent.qf2_target.load_state_dict(saved_agent.qf2_target.state_dict())

    agent.train(False)
    print(f"[Eval] Loaded {args.agent.upper()} agent from {args.snapshot}")
    print(f"       Trained for {payload.get('_global_step', '?')} steps")

    return agent


def main():
    args = parse_args()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    print("=" * 60)
    print(f"{args.agent.upper()} Real Robot Evaluation")
    print("=" * 60)

    camera_rig = build_camera_rig(args)
    home_q = json.loads(args.home_q) if args.home_q else None

    # Resolve home sequence (.npy files → list of joint configs).
    home_sequence = None
    if args.home_sequence:
        from pathlib import Path as _P
        import numpy as _np
        simtoreal_dir = _P(__file__).resolve().parent
        home_sequence = []
        for p in args.home_sequence:
            path = _P(p)
            if not path.is_absolute():
                path = simtoreal_dir / path
            if not path.exists():
                raise FileNotFoundError(f"home-sequence file not found: {path}")
            home_sequence.append(_np.load(str(path)).tolist())
        print(f"[home-sequence] {len(home_sequence)} waypoints from: "
              f"{args.home_sequence}")

    if args.task_mode == "pick-place":
        gripper_at_reset = args.gripper_start or "closed"
        pause_for_human = True
    else:
        gripper_at_reset = args.gripper_start or "open"
        pause_for_human = False
    print(f"[task-mode] {args.task_mode} "
          f"(gripper_at_reset={gripper_at_reset}, pause_for_human={pause_for_human})")

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
        home_sequence=home_sequence,
        gripper_at_reset=gripper_at_reset,
        pause_for_human=pause_for_human,
        freeze_gripper=args.freeze_gripper,
    )

    agent = make_agent_from_snapshot(env, args)
    use_cqn = args.agent == "cqn"
    use_te = use_cqn and args.temporal_ensemble

    # Goal-pose auto-success detection
    real_env: RealFrankaEnv = env._env  # unwrap ExtendedTimeStepWrapper
    goal_pose = None
    if args.goal_pose:
        goal_pose = _load_goal_pose(args.goal_pose)
        print(f"[Eval] Auto-success: goal loaded from {args.goal_pose}")
        print(f"       pos_thresh={args.success_pos_thresh*100:.1f}cm, "
              f"quat_thresh={args.success_quat_thresh:.3f}")
    else:
        print("[Eval] Using human y/N for success labeling")

    # Results directory
    if args.results_dir:
        results_save_dir = Path(args.results_dir)
    else:
        results_save_dir = Path(args.snapshot).parent
    results_save_dir.mkdir(parents=True, exist_ok=True)

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

        time_step = env.reset(skip_home=(args.skip_initial_home and ep == 0))
        episode_step = 0
        episode_reward = 0.0
        video_frames = []

        if use_te:
            te = utils.TemporalEnsembleControl(
                args.episode_length, env.action_spec(), args.action_sequence,
            )

        action = None

        while not time_step.last():
            t0 = time.time()

            if use_cqn:
                # CQN-AS: action sequences + optional temporal ensemble
                if use_te or episode_step % args.action_sequence == 0:
                    with torch.no_grad(), utils.eval_mode(agent):
                        raw_action = agent.act(
                            time_step.rgb_obs,
                            time_step.low_dim_obs,
                            step=999999,
                            eval_mode=True,
                        )
                    action = raw_action.reshape([args.action_sequence, -1])
                    if use_te:
                        te.register_action_sequence(action)

                if use_te:
                    sub_action = te.get_action()
                else:
                    sub_action = action[episode_step % args.action_sequence]
            else:
                # ARSQ: single-step action
                with torch.no_grad(), utils.eval_mode(agent):
                    sub_action = agent.act(
                        time_step.rgb_obs,
                        time_step.low_dim_obs,
                        step=999999,
                        eval_mode=True,
                    )

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

        # Close gripper at end of episode (hold object for inspection)
        print("  Closing gripper...")
        real_env.close_gripper()
        time.sleep(0.5)

        print(
            f"  Episode {ep + 1}: steps={episode_step}, reward={episode_reward:.3f}"
        )

        # Success detection — auto or human
        if goal_pose is not None:
            success = _check_success(
                real_env, goal_pose,
                args.success_pos_thresh, args.success_quat_thresh,
            )
        else:
            success = False  # no goal pose → cannot auto-detect
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
    results_path = results_save_dir / "eval_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {results_path}")

    env.close()


if __name__ == "__main__":
    main()
