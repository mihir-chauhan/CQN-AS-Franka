#!/bin/bash
#SBATCH --job-name=open_oven
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --gpus-per-node=1
#SBATCH --time=12:00:00
#SBATCH --partition=smallgpu
#SBATCH -A lilly-rob1
#SBATCH --output=slurm-logs/slurm-%j-open_oven.out
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=hviswan@purdue.edu

# ==========================================
# 0. GLOBAL ENVIRONMENT SETUP
# ==========================================
echo "Setting up environment..."
module load conda
source activate /depot/bera89/data/hviswan/cqn_as

# 1. Setup Conda Paths for Xvfb and System Libs
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib
export PATH=$CONDA_PREFIX/x86_64-conda-linux-gnu/sysroot/usr/bin:$PATH

# 2. Fix libcrypto for Xvfb
if [ ! -L "$CONDA_PREFIX/lib/libcrypto.so.10" ]; then
    ln -sf $CONDA_PREFIX/lib/libcrypto.so.1.0.0 $CONDA_PREFIX/lib/libcrypto.so.10
fi
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# 3. Setup CoppeliaSim Paths (REQUIRED for both Data Gen and Training)
export COPPELIASIM_ROOT=/home/hviswan/CoppeliaSim
export LD_LIBRARY_PATH=$COPPELIASIM_ROOT:$LD_LIBRARY_PATH
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT

# 4. Generate Unique Display ID
# This prevents conflicts if multiple jobs land on the same node
export UNIQUE_DISPLAY_ID=$((100 + ($SLURM_JOB_ID % 200)))
echo "Environment Ready. Using Display ID: $UNIQUE_DISPLAY_ID"

# ==========================================
# PHASE 1: DATASET GENERATION
# ==========================================
cd /scratch/gautschi/hviswan/CQN-AS/RLBench/rlbench
echo "Starting Dataset Generation for open_drawer..."

# Start Xvfb
# We use the flags +extension GLX +render to ensure OpenGL works
Xvfb :$UNIQUE_DISPLAY_ID -screen 0 1280x1024x24 +extension GLX +render > xvfb_gen_open_oven.log 2>&1 &
PID_XVFB=$!
sleep 5
export DISPLAY=:$UNIQUE_DISPLAY_ID

# Check if Xvfb is alive
if ! kill -0 $PID_XVFB 2>/dev/null; then
    echo "CRITICAL: Xvfb failed to start for Data Generation."
    cat xvfb_gen_open_oven.log
    exit 1
fi

# Run Generator
python dataset_generator.py \
  --save_path=/scratch/gautschi/hviswan/CQN-AS/dataset/ \
  --image_size 84 84 \
  --renderer opengl3 \
  --episodes_per_task 100 \
  --variations 1 \
  --processes 1 \
  --tasks open_oven \
  --arm_max_velocity 2.0 \
  --arm_max_acceleration 8.0

# Cleanup Phase 1
kill $PID_XVFB
rm -f /tmp/.X$UNIQUE_DISPLAY_ID-lock
echo "Dataset Generation Complete."

# ==========================================
# PHASE 2: TRAINING
# ==========================================
cd /scratch/gautschi/hviswan/ARSQ
task="open_drawer"
echo "Starting Training for open_drawer..."

# Start Xvfb (Phase 2)
Xvfb :$UNIQUE_DISPLAY_ID -screen 0 1024x768x24 +extension GLX +render > xvfb_train_open_oven.log 2>&1 &
XVFB_PID=$!
sleep 5
export DISPLAY=:$UNIQUE_DISPLAY_ID

# Check if Xvfb is alive
if ! kill -0 $XVFB_PID 2>/dev/null; then
    echo "CRITICAL: Xvfb failed to start for Training."
    cat xvfb_train_open_oven.log
    exit 1
fi

# Run Trainer
# Using 'task_name' and explicit WandB entity
# python train_cqn_as_rlbench.py \
#     task_name=open_oven \
#     num_demos=100 \
#     dataset_root=/scratch/gautschi/hviswan/CQN-AS/dataset \
#     wandb.name=exp_open_oven_fk \
#     wandb.entity=hrishi-vish \
#     wandb.project=cqn
cd rlbench
python arsq_rlb/runner/train_rlbench_sqar.py \
   dataset_root=/scratch/gautschi/hviswan/CQN-AS/dataset \
   rlbench_task=${task} \
   wandb.name="arsq-${task}" \
   seed=0

# Final Cleanup
kill $XVFB_PID
rm -f /tmp/.X$UNIQUE_DISPLAY_ID-lock
echo "Job Complete."
