#!/bin/bash
#SBATCH --job-name=rlb_bask_lamp
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --gpus-per-node=1
#SBATCH --time=12:00:00
#SBATCH --partition=smallgpu
#SBATCH -A lilly-rob1
#SBATCH --output=slurm-logs/slurm-%j-%x.out
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=hviswan@purdue.edu

# ==========================================
# 0. GLOBAL ENVIRONMENT SETUP
# ==========================================
echo "Setting up environment..."
module load conda
source activate /depot/bera89/data/hviswan/cqn_as

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib
export PATH=$CONDA_PREFIX/x86_64-conda-linux-gnu/sysroot/usr/bin:$PATH

if [ ! -L "$CONDA_PREFIX/lib/libcrypto.so.10" ]; then
    ln -sf $CONDA_PREFIX/lib/libcrypto.so.1.0.0 $CONDA_PREFIX/lib/libcrypto.so.10
fi
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

export COPPELIASIM_ROOT=/home/hviswan/CoppeliaSim
export LD_LIBRARY_PATH=$COPPELIASIM_ROOT:$LD_LIBRARY_PATH
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT

export UNIQUE_DISPLAY_ID=$((100 + ($SLURM_JOB_ID % 200)))
echo "Environment Ready. Using Display ID: $UNIQUE_DISPLAY_ID"

# ==========================================
# REUSABLE PIPELINE FUNCTION
# ==========================================
run_rl_pipeline() {
    local TARGET_TASK=$1
    echo "====================================================="
    echo " STARTING PIPELINE FOR: ${TARGET_TASK}"
    echo "====================================================="

    # --- PHASE 1: DATASET GENERATION ---
    cd /scratch/gautschi/hviswan/CQN-AS/RLBench/rlbench
    echo "Phase 1: Generating Data for ${TARGET_TASK}..."

    Xvfb :$UNIQUE_DISPLAY_ID -screen 0 1280x1024x24 +extension GLX +render > "xvfb_gen_${TARGET_TASK}.log" 2>&1 &
    PID_XVFB=$!
    sleep 5
    export DISPLAY=:$UNIQUE_DISPLAY_ID

    if ! kill -0 $PID_XVFB 2>/dev/null; then
        echo "CRITICAL: Xvfb failed to start for Data Gen (${TARGET_TASK})."
        cat "xvfb_gen_${TARGET_TASK}.log"
        exit 1
    fi

    python dataset_generator.py \
      --save_path=/scratch/gautschi/hviswan/CQN-AS/dataset/ \
      --image_size 84 84 \
      --renderer opengl3 \
      --episodes_per_task 100 \
      --variations 1 \
      --processes 1 \
      --tasks ${TARGET_TASK} \
      --arm_max_velocity 2.0 \
      --arm_max_acceleration 8.0

    kill $PID_XVFB
    rm -f /tmp/.X$UNIQUE_DISPLAY_ID-lock
    echo "Phase 1 Complete for ${TARGET_TASK}."

    # --- PHASE 2: TRAINING ---
    cd /scratch/gautschi/hviswan/ARSQ/rlbench
    echo "Phase 2: Training for ${TARGET_TASK}..."

    Xvfb :$UNIQUE_DISPLAY_ID -screen 0 1024x768x24 +extension GLX +render > "xvfb_train_${TARGET_TASK}.log" 2>&1 &
    XVFB_PID=$!
    sleep 5
    export DISPLAY=:$UNIQUE_DISPLAY_ID

    if ! kill -0 $XVFB_PID 2>/dev/null; then
        echo "CRITICAL: Xvfb failed to start for Training (${TARGET_TASK})."
        cat "xvfb_train_${TARGET_TASK}.log"
        exit 1
    fi

    # python arsq_rlb/runner/train_rlbench_sqar.py \
    #    dataset_root=/scratch/gautschi/hviswan/CQN-AS/dataset \
    #    rlbench_task=${TARGET_TASK} \
    #    wandb.name="arsq-NOPINN_${TARGET_TASK}_CQN_AS" \
    #    seed=0
    MODEL="sqar"
    USE_EIKONAL="False"
    USE_FK="True"
    if [ "$REG_TYPE" == "eikonal" ]; then
        USE_EIKONAL="True"
    elif [ "$REG_TYPE" == "fk" ]; then
        USE_FK="True"
    fi

    WANDB_RUN_NAME="${MODEL}-${REG_TYPE}_${TARGET_TASK}"
    python arsq_rlb/runner/train_rlbench_sqar.py --config-name=config_rlbench_${MODEL} \
        dataset_root=/scratch/gautschi/hviswan/CQN-AS/dataset \
        rlbench_task=${TARGET_TASK} \
        agent.use_eikonal=${USE_EIKONAL} \
        agent.use_fk_reg=${USE_FK} \
        wandb.name="${WANDB_RUN_NAME}" \
        seed=0
    kill $XVFB_PID
    rm -f /tmp/.X$UNIQUE_DISPLAY_ID-lock
    echo "Phase 2 Complete for ${TARGET_TASK}."
}

# ==========================================
# EXECUTE TASKS
# ==========================================
#run_rl_pipeline "basketball_in_hoop"
run_rl_pipeline "lamp_on"

echo "All tasks for this job completed successfully."
