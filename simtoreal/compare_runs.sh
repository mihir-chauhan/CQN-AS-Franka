#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════
# compare_runs.sh — Launch 3 baseline training runs for comparison
#
# All 3 runs use the SAME demos, camera setup, and hyperparams.
# The only difference is the agent / physics regularizer.
#
# EDIT the variables below to match your setup, then run each
# command one at a time (they use the real robot sequentially).
# ═══════════════════════════════════════════════════════════════════
set -euo pipefail

# ── Shared config — EDIT THESE ──────────────────────────────────
ROBOT_IP="192.168.131.41"
CAMERA_MODE="orbbec"
ORBBEC='{"front":"CL25854009Y","wrist":"CL2E453000Y","left_shoulder":"CL2E4530083","right_shoulder":"CL25854007B"}'

DEMO_DIR="../runs/shelf_4cam/demos"       # existing recorded demos
GOAL_POSE="goal_pose.json"                # for auto success detection

NUM_TRAIN_STEPS=10000
EPISODE_LENGTH=200
BATCH_SIZE=32
DEMO_BATCH_SIZE=32
EVAL_EVERY=2000
SAVE_EVERY=500

# ── Common flags ────────────────────────────────────────────────
COMMON=(
    --robot-ip "$ROBOT_IP"
    --camera-mode "$CAMERA_MODE"
    --orbbec-serials "$ORBBEC"
    --load-demos-only
    --demo-dir "$DEMO_DIR"
    --goal-pose "$GOAL_POSE"
    --num-train-steps "$NUM_TRAIN_STEPS"
    --episode-length "$EPISODE_LENGTH"
    --batch-size "$BATCH_SIZE"
    --demo-batch-size "$DEMO_BATCH_SIZE"
    --eval-every "$EVAL_EVERY"
    --save-every "$SAVE_EVERY"
)

echo "═══════════════════════════════════════════════════════════"
echo "  3-Way Comparison: CQN-AS vs ARSQ-base vs ARSQ+physics"
echo "  Demos: $DEMO_DIR"
echo "  Steps: $NUM_TRAIN_STEPS"
echo "═══════════════════════════════════════════════════════════"

# ─────────────────────────────────────────────────────────────────
# RUN 1: CQN-AS (distributional RL + action sequences)
# ─────────────────────────────────────────────────────────────────
echo ""
echo "══ RUN 1: CQN-AS ══════════════════════════════════════════"
echo "  Save: ../runs/compare_cqn_as"
echo "  Log:  ../runs/compare_cqn_as/train.log"
echo ""
echo "  To start:"
echo "    python train_real.py \\"
echo "        --agent cqn \\"
echo "        ${COMMON[*]} \\"
echo "        --save-dir ../runs/compare_cqn_as \\"
echo "        2>&1 | tee ../runs/compare_cqn_as/train.log"
echo ""

# ─────────────────────────────────────────────────────────────────
# RUN 2: ARSQ-base (no physics regularization)
# ─────────────────────────────────────────────────────────────────
echo "══ RUN 2: ARSQ-base (no physics) ═════════════════════════"
echo "  Save: ../runs/compare_arsq_base"
echo "  Log:  ../runs/compare_arsq_base/train.log"
echo ""
echo "  To start:"
echo "    python train_real.py \\"
echo "        --agent arsq --no-fk-reg \\"
echo "        ${COMMON[*]} \\"
echo "        --save-dir ../runs/compare_arsq_base \\"
echo "        2>&1 | tee ../runs/compare_arsq_base/train.log"
echo ""

# ─────────────────────────────────────────────────────────────────
# RUN 3: ARSQ + Physics (FK Walk-on-Spheres regularization)
# ─────────────────────────────────────────────────────────────────
echo "══ RUN 3: ARSQ + Physics (FK reg) ════════════════════════"
echo "  Save: ../runs/compare_arsq_phys"
echo "  Log:  ../runs/compare_arsq_phys/train.log"
echo ""
echo "  To start:"
echo "    python train_real.py \\"
echo "        --agent arsq --use-fk-reg --fk-weight 1.0 \\"
echo "        --kappa 0.1 --nu 0.01 --num-walks 10 \\"
echo "        ${COMMON[*]} \\"
echo "        --save-dir ../runs/compare_arsq_phys \\"
echo "        2>&1 | tee ../runs/compare_arsq_phys/train.log"
echo ""

echo "═══════════════════════════════════════════════════════════"
echo "After all 3 runs finish, compare with:"
echo ""
echo "  python plot_comparison.py \\"
echo "      --labels 'CQN-AS' 'ARSQ-base' 'ARSQ+Physics' \\"
echo "      --logs ../runs/compare_cqn_as/train.log \\"
echo "           ../runs/compare_arsq_base/train.log \\"
echo "           ../runs/compare_arsq_phys/train.log \\"
echo "      --save comparison.png"
echo "═══════════════════════════════════════════════════════════"
