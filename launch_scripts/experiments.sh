#!/bin/bash

# Exit on any error
set -e

# Parse command line arguments
skip_init=false
skip_finetune=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-init)
            skip_init=true
            shift
            ;;
        --skip-finetune)
            skip_finetune=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--skip-init] [--skip-finetune]"
            exit 1
            ;;
    esac
done

# Create necessary directories
mkdir -p results
mkdir -p data
mkdir -p logs

# Get current timestamp for log file
timestamp=$(date +"%Y%m%d_%H%M%S")
log_file="logs/experiment_${timestamp}.log"

# Function to log messages to both console and file
log() {
    echo "$1" | tee -a "$log_file"
}

# Function to run command with logging
run_cmd() {
    log "Running: $1"
    eval "$1" 2>&1 | tee -a "$log_file"
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        log "Error: Command failed"
        exit 1
    fi
}

# Start logging
log "=== Starting experiment at $(date) ==="

# Common arguments used across commands
COMMON_ARGS="--data-location ./data \
    --save ./results \
    --model ViT-B-32 \
    --batch-size 32 \
    --lr 1e-4 \
    --wd 0.0"

# 0. Initialize pre-trained model (if not skipped)
if [ "$skip_init" = false ]; then
    log "\n=== Initializing pre-trained model ==="
    run_cmd "python init_pretrained.py $COMMON_ARGS"
fi

# 1. Fine-tune on all datasets (if not skipped)
if [ "$skip_finetune" = false ]; then
    log "\n=== Fine-tuning on all datasets ==="
    run_cmd "python finetune.py $COMMON_ARGS"
fi

# 2. Evaluate single-task performance
log "\n=== Evaluating single-task performance ==="
run_cmd "python eval_single_task.py $COMMON_ARGS"

# 3. Perform task addition
log "\n=== Performing task addition ==="
run_cmd "python eval_task_addition.py $COMMON_ARGS"

# Clean up intermediate files to save space
log "\n=== Cleaning up to save space ==="
find ./results -name "*.pt" ! -name "pretrained.pt" ! -name "*_finetuned.pt" -delete

log "\n=== Experiment completed at $(date) ==="
