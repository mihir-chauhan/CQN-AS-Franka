#!/bin/bash
#SBATCH --job-name=rlb_data_gen
#SBATCH --nodes=1
#SBATCH --ntasks=64                  # Adjust cores if data gen doesn't need all 64
#SBATCH --gpus-per-node=1            
#SBATCH --time=12:00:00              # Adjust time based on how long 900 total episodes take
#SBATCH --partition=smallgpu
#SBATCH -A lilly-rob1
#SBATCH --output=slurm-logs/data_gen_%j.out  
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hviswan@purdue.edu

# ==========================================
# 1. GLOBAL ENVIRONMENT SETUP
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

export UNIQUE_DISPLAY_ID=$((100 + ($SLURM_JOB_ID % 100)))

TASKS=(
  "take_lid_off_saucepan" "turn_tap" "sweep_to_dustpan" 
  "stack_wine" "slide_block_to_target" "reach_target" 
  "pick_up_cup" "meat_on_grill" "insert_usb_in_computer"
)

# ==========================================
# 2. SEQUENTIAL DATASET GENERATION
# ==========================================
cd /scratch/gautschi/hviswan/CQN-AS/RLBench/rlbench

for TARGET_TASK in "${TASKS[@]}"; do
    echo "====================================================="
    echo " GENERATING DATASET FOR: ${TARGET_TASK}"
    echo "====================================================="

    DATASET_DIR="/scratch/gautschi/hviswan/CQN-AS/dataset/${TARGET_TASK}"
    
    if [ -d "$DATASET_DIR" ]; then
        echo "Dataset for ${TARGET_TASK} already exists. Skipping..."
        continue
    fi

    Xvfb :$UNIQUE_DISPLAY_ID -screen 0 1280x1024x24 +extension GLX +render > "xvfb_gen_${TARGET_TASK}.log" 2>&1 &
    PID_XVFB=$!
    sleep 5
    export DISPLAY=:$UNIQUE_DISPLAY_ID

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

    # Clean up generator display before the next task loop
    kill -9 $PID_XVFB 2>/dev/null
    rm -f /tmp/.X$UNIQUE_DISPLAY_ID-lock
    echo "Finished generating ${TARGET_TASK}."
done

echo "ALL DATASETS GENERATED SUCCESSFULLY."