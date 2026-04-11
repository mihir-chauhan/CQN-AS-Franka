#!/bin/bash
#SBATCH --job-name=rlb_ablations
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
    local MODEL=$2       # "sqar" or "cqnas"
    local REG_TYPE=$3    # "eikonal" or "fk"
    
    echo "====================================================="
    echo " STARTING PIPELINE:"
    echo " TASK: ${TARGET_TASK} | MODEL: ${MODEL} | REG: ${REG_TYPE}"
    echo "====================================================="

    # --- Enforce Mutual Exclusivity for Regularizers ---
    local USE_EIKONAL="False"
    local USE_FK="False"

    if [ "$REG_TYPE" == "eikonal" ]; then
        USE_EIKONAL="True"
    elif [ "$REG_TYPE" == "fk" ]; then
        USE_FK="True"
    else
        echo "Error: Unknown regularization type '$REG_TYPE'. Use 'eikonal' or 'fk'."
        exit 1
    fi

    # --- Construct Dynamic WandB Name ---
    local WANDB_RUN_NAME="${MODEL}-${REG_TYPE}_${TARGET_TASK}"

    # --- PHASE 2: TRAINING ---
    cd /scratch/gautschi/hviswan/ARSQ/rlbench
    echo "Phase 2: Training..."

    Xvfb :$UNIQUE_DISPLAY_ID -screen 0 1024x768x24 +extension GLX +render > "xvfb_train_${WANDB_RUN_NAME}.log" 2>&1 &
    XVFB_PID=$!
    sleep 5
    export DISPLAY=:$UNIQUE_DISPLAY_ID

    if ! kill -0 $XVFB_PID 2>/dev/null; then
        echo "CRITICAL: Xvfb failed to start for Training."
        cat "xvfb_train_${WANDB_RUN_NAME}.log"
        exit 1
    fi

    # Execute training with dynamic config override and strict flags
    python arsq_rlb/runner/train_rlbench_sqar.py --config-name=config_rlbench_${MODEL} \
        dataset_root=/scratch/gautschi/hviswan/CQN-AS/dataset \
        rlbench_task=${TARGET_TASK} \
        agent.use_eikonal=${USE_EIKONAL} \
        agent.use_fk_reg=${USE_FK} \
        wandb.name="${WANDB_RUN_NAME}" \
        seed=0

    kill $XVFB_PID
    rm -f /tmp/.X$UNIQUE_DISPLAY_ID-lock
    echo "Training Complete for ${WANDB_RUN_NAME}."
}

# ==========================================
# EXECUTE 4 ABLATION TASKS SEQUENTIALLY
# ==========================================

# Ablation 1: SQAR with exact Autograd Eikonal
run_rl_pipeline "press_switch" "sqar" "eikonal"

# Ablation 2: SQAR with Walk-on-Spheres FK
run_rl_pipeline "press_switch" "sqar" "fk"

# Ablation 3: CQN_AS with exact Autograd Eikonal
run_rl_pipeline "press_switch" "cqnas" "eikonal"

# Ablation 4: CQN_AS with Walk-on-Spheres FK
run_rl_pipeline "press_switch" "cqnas" "fk"