#!/bin/bash

# Function to submit a job and wait for it to finish
submit_and_wait() {
    local SCRIPT=$1
    local OFFSET=$2
    
    # Submit the job and capture the output (e.g., "Submitted batch job 123456")
    local OUTPUT=$(sbatch $SCRIPT $OFFSET)
    local JOB_ID=$(echo $OUTPUT | awk '{print $4}')
    
    echo "=========================================================="
    echo "$(date): Submitted $SCRIPT with Offset $OFFSET -> Slurm Job ID $JOB_ID"
    echo "Waiting for completion (~6 hours)..."
    echo "=========================================================="
    
    # Give Slurm 30 seconds to register the job in the queue
    sleep 30 
    
    # Loop continuously as long as the Job ID appears in squeue
    while squeue -j $JOB_ID 2>/dev/null | grep -q $JOB_ID; do
        sleep 300 # Check the queue every 5 minutes
    done
    
    echo "$(date): Job $JOB_ID completed! Moving to next batch..."
}

# # ---------------------------------------------------------
# # PHASE 1: GENERATE DATASETS (9 Tasks total)
# # Offsets: 0, 3, 6
# # ---------------------------------------------------------
# echo "STARTING DATASET GENERATION PHASE..."
# for OFFSET in 0 3 6; do
#     submit_and_wait "data_gen.sh" $OFFSET
# done

# ---------------------------------------------------------
# PHASE 2: TRAIN MODELS (36 Combinations total)
# Offsets: 0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33
# ---------------------------------------------------------
echo "STARTING TRAINING PHASE..."
for OFFSET in {0..33..3}; do
    submit_and_wait "train_ablations.sh" $OFFSET
done

echo "ALL DATASETS GENERATED AND ALL MODELS TRAINED SUCCESSFULLY."