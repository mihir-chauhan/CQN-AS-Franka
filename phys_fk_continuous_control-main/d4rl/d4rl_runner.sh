#!/bin/bash
#SBATCH --job-name=d4rl_train_matrix
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
  "hm" "hmr" "wm" "cm" "cme"
)

# Reduced to 4 ablations: eikonal & fk for both SQAR and CQN
MODELS=(   "sqar"    "sqar"  "cqn"     "cqn"  )
EIKONALS=( "True"    "False" "True"    "False")
FKS=(      "False"   "True"  "False"   "True" )
REG_NAMES=("eikonal" "fk"    "eikonal" "fk"   )

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

export LD_LIBRARY_PATH=$HOME/.mujoco/mujoco210/bin:$LD_LIBRARY_PATH

export D4RL_DATASET_DIR=$(pwd)/datasets_${WANDB_RUN_NAME}
# conda env config vars set MUJOCO_PY_MUJOCO_PATH=$MUJOCO_PY_MUJOCO_PATH
# conda env config vars set LD_LIBRARY_PATH=$LD_LIBRARY_PATH
# conda env config vars set D4RL_DATASET_DIR=$D4RL_DATASET_DIR
export PYTHONPATH="/scratch/gautschi/hviswan/ARSQ/d4rl:$PYTHONPATH"

export UNIQUE_DISPLAY_ID=$((100 + ($SLURM_JOB_ID % 100) + $SLURM_ARRAY_TASK_ID))

# ==========================================
# 4. START VIRTUAL DISPLAY & TRAINING
# ==========================================
cd /scratch/gautschi/hviswan/ARSQ/d4rl

Xvfb :$UNIQUE_DISPLAY_ID -screen 0 1024x768x24 +extension GLX +render > "xvfb_train_${WANDB_RUN_NAME}.log" 2>&1 &
XVFB_PID=$!
sleep 5
export DISPLAY=:$UNIQUE_DISPLAY_ID

# ==========================================
# 5. EXECUTION 
# ==========================================
CUDA_VISIBLE_DEVICES=0 python arsq_d4rl/run/${MODEL}_run.py \
    env_cfg@_global_=${TARGET_TASK} \
    alg.use_eikonal=${USE_EIKONAL} \
    alg.use_fk_reg=${USE_FK} \
    wandb.name="${WANDB_RUN_NAME}" \
    seed=0

kill -9 $XVFB_PID 2>/dev/null
rm -f /tmp/.X$UNIQUE_DISPLAY_ID-lock
echo "Training Complete for ${WANDB_RUN_NAME}."