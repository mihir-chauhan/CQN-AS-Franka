#!/bin/bash
#SBATCH --job-name=rlb_train_matrix
#SBATCH --array=0-2                  # Queue exactly 3 jobs at a time
#SBATCH --nodes=1
#SBATCH --ntasks=64                  
#SBATCH --gpus-per-node=1            
#SBATCH --time=12:00:00
#SBATCH --partition=smallgpu
#SBATCH -A lilly-rob1
#SBATCH --output=slurm-logs/slurm-%A_%a.out  
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hviswan@purdue.edu

# ==========================================
# 1. HANDLE QUEUE OFFSET
# ==========================================
OFFSET=${1:-0} 
INDEX=$((SLURM_ARRAY_TASK_ID + OFFSET))

# ==========================================
# 2. DEFINE THE MASTER MATRIX
# ==========================================
TASKS=(
  "take_lid_off_saucepan" "turn_tap" "sweep_to_dustpan" 
  "stack_wine" "slide_block_to_target" "reach_target" 
  "pick_up_cup" "meat_on_grill" "insert_usb_in_computer"
)

MODELS=(   "sqar"  "sqar"  "sqar"  "cqnas" )
EIKONALS=( "False" "True"  "False" "False" )
FKS=(      "False" "False" "True"  "False" )
REG_NAMES=("base"  "eikonal" "fk"  "base"  )

TASK_IDX=$((INDEX / 4))
ABL_IDX=$((INDEX % 4))

if [ "$TASK_IDX" -ge "${#TASKS[@]}" ]; then
    echo "Global Index $INDEX out of bounds. All training completed."
    exit 0
fi

TARGET_TASK=${TASKS[$TASK_IDX]}
MODEL=${MODELS[$ABL_IDX]}
USE_EIKONAL=${EIKONALS[$ABL_IDX]}
USE_FK=${FKS[$ABL_IDX]}
REG_NAME=${REG_NAMES[$ABL_IDX]}

WANDB_RUN_NAME="${MODEL}-${REG_NAME}_${TARGET_TASK}"

echo "====================================================="
echo " TRAINING WORKER: Array ID ${SLURM_ARRAY_TASK_ID} | Global Index ${INDEX}"
echo " TASK: ${TARGET_TASK} | MODEL: ${MODEL} | EIKONAL: ${USE_EIKONAL} | FK: ${USE_FK}"
echo "====================================================="

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

export UNIQUE_DISPLAY_ID=$((100 + ($SLURM_JOB_ID % 100) + $SLURM_ARRAY_TASK_ID))

# ==========================================
# 4. START VIRTUAL DISPLAY & TRAINING
# ==========================================
cd /scratch/gautschi/hviswan/ARSQ/rlbench

Xvfb :$UNIQUE_DISPLAY_ID -screen 0 1024x768x24 +extension GLX +render > "xvfb_train_${WANDB_RUN_NAME}.log" 2>&1 &
XVFB_PID=$!
sleep 5
export DISPLAY=:$UNIQUE_DISPLAY_ID

if ! kill -0 $XVFB_PID 2>/dev/null; then
    echo "CRITICAL: Xvfb failed to start for Training."
    exit 1
fi

python arsq_rlb/runner/train_rlbench_sqar.py --config-name=config_rlbench_${MODEL} \
    dataset_root=/scratch/gautschi/hviswan/CQN-AS/dataset \
    rlbench_task=${TARGET_TASK} \
    agent.use_eikonal=${USE_EIKONAL} \
    agent.use_fk_reg=${USE_FK} \
    wandb.name="${WANDB_RUN_NAME}" \
    seed=0

kill -9 $XVFB_PID 2>/dev/null
rm -f /tmp/.X$UNIQUE_DISPLAY_ID-lock
echo "Training Complete for ${WANDB_RUN_NAME}."