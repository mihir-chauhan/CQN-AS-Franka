#!/bin/bash

submit_and_wait() {
    local SCRIPT=$1
    local OFFSET=$2
    
    local OUTPUT=$(sbatch $SCRIPT $OFFSET)
    local JOB_ID=$(echo $OUTPUT | awk '{print $4}')
    
    echo "=========================================================="
    echo "$(date): Submitted $SCRIPT with Offset $OFFSET -> Slurm Job ID $JOB_ID"
    echo "Waiting for completion (~12 hours max)..."
    echo "=========================================================="
    
    sleep 30 
    
    while squeue -j $JOB_ID 2>/dev/null | grep -q $JOB_ID; do
        sleep 300 
    done
    
    echo "$(date): Job $JOB_ID completed! Moving to next batch..."
}

echo "STARTING D4RL MATRIX TRAINING (20 runs total in batches of 3)..."

# Loop from 0 to 18, stepping by 3
for OFFSET in {0..18..3}; do
    submit_and_wait "d4rl_runner.sh" $OFFSET
done

echo "ALL D4RL MODELS TRAINED SUCCESSFULLY."