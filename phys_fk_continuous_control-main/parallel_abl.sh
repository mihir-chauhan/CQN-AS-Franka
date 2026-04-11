#!/bin/bash
#SBATCH --job-name=rlb_ablation_array
#SBATCH --array=0-3                  # Creates 4 parallel jobs, indexed 0 through 3
#SBATCH --nodes=1
#SBATCH --ntasks=64                  # Note: Each of the 4 jobs will request 64 cores
#SBATCH --gpus-per-node=1            # Each of the 4 jobs gets 1 GPU
#SBATCH --time=12:00:00
#SBATCH --partition=smallgpu
#SBATCH -A lilly-rob1
#SBATCH --output=slurm-logs/slurm-%A_%a.out  # %A is the base Job ID, %a is the Array ID
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hviswan@purdue.edu

# ==========================================
# 1. DEFINE THE ABLATION MATRIX
# ==========================================
# Indices:      0         1         2         3
MODELS=(      "sqar"    "sqar"    "cqnas"   "cqnas"  )
REG_TYPES=(   "eikonal" "fk"      "eikonal" "fk"     )

# Fetch the exact configuration for this specific parallel worker
TARGET_TASK="open_drawer"
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}
REG_TYPE=${REG_TYPES[$SLURM_ARRAY_TASK_ID]}

echo "====================================================="
echo " STARTING PARALLEL WORKER: Array Task ID ${SLURM_ARRAY_TASK_ID}"
echo " TASK: ${TARGET_TASK} | MODEL: ${MODEL} | REG: ${REG_TYPE}"
echo "====================================================="

# ==========================================
# 2. RESOLVE REGULARIZER FLAGS
# ==========================================
USE_EIKONAL="False"
USE_FK="False"

if [ "$REG_TYPE" == "eikonal" ]; then
    USE_EIKONAL="True"
elif [ "$REG_TYPE" == "fk" ]; then
    USE_FK="True"
fi

WANDB_RUN_NAME="${MODEL}-${REG_TYPE}_${TARGET_TASK}"

# ==========================================
# 3. GLOBAL ENVIRONMENT SETUP
# ==========================================
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

# CRITICAL: Ensure unique Xvfb display IDs for parallel jobs on the same node
export UNIQUE_DISPLAY_ID=$((100 + ($SLURM_JOB_ID % 100) + $SLURM_ARRAY_TASK_ID))
echo "Environment Ready. Using Display ID: $UNIQUE_DISPLAY_ID"

# ==========================================
# 4. START VIRTUAL DISPLAY & TRAINING
# ==========================================
cd /scratch/gautschi/hviswan/ARSQ/rlbench
echo "Starting Xvfb for ${WANDB_RUN_NAME}..."

Xvfb :$UNIQUE_DISPLAY_ID -screen 0 1024x768x24 +extension GLX +render > "xvfb_train_${WANDB_RUN_NAME}.log" 2>&1 &
XVFB_PID=$!
sleep 5
export DISPLAY=:$UNIQUE_DISPLAY_ID

if ! kill -0 $XVFB_PID 2>/dev/null; then
    echo "CRITICAL: Xvfb failed to start for Training."
    cat "xvfb_train_${WANDB_RUN_NAME}.log"
    exit 1
fi

echo "Launching Python training script..."
python arsq_rlb/runner/train_rlbench_sqar.py --config-name=config_rlbench_${MODEL} \
    dataset_root=/scratch/gautschi/hviswan/CQN-AS/dataset \
    rlbench_task=${TARGET_TASK} \
    agent.use_eikonal=${USE_EIKONAL} \
    agent.use_fk_reg=${USE_FK} \
    wandb.name="${WANDB_RUN_NAME}" \
    seed=0

# Clean up
kill $XVFB_PID
rm -f /tmp/.X$UNIQUE_DISPLAY_ID-lock
echo "Training Complete for ${WANDB_RUN_NAME}."