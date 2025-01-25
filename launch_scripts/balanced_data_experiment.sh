#!/bin/bash

set -e

mkdir -p balanced_data_results
mkdir -p logs

timestamp=$(date +"%Y%m%d_%H%M%S")
log_file="logs/balanced_data_experiment_${timestamp}.log"

log() {
    echo "$1" | tee -a "$log_file"
}

run_cmd() {
    log "Running: $1"
    eval "$1" 2>&1 | tee -a "$log_file"
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        log "Error: Command failed"
        exit 1
    fi
}

run_experiment() {
    local name=$1
    local lr=$2
    local wd=$3
    local script=${4:-finetune.py}
    
    log "\n=== $name configuration ==="
    run_cmd "python $script $BASE_ARGS --save ./balanced_data_results/$name --lr $lr --wd $wd"
    run_cmd "python eval_single_task.py $BASE_ARGS --save ./balanced_data_results/$name --lr $lr --wd $wd"
    run_cmd "python eval_task_addition.py $BASE_ARGS --save ./balanced_data_results/$name --lr $lr --wd $wd"
}

if [ "$1" = "launcher" ]; then
    # This is the launcher mode that starts both jobs in parallel
    log "=== Starting all jobs in parallel at $(date) ==="
    
    # Start job 1 in background
    $0 1 &
    PID1=$!
    
    # Start job 2 in background
    $0 2 &
    PID2=$!
    
    # Wait for both jobs to complete
    wait $PID1 $PID2
    
    log "=== All jobs completed at $(date) ==="
    exit 0
fi

# Normal job execution code
log "=== Starting balanced data experiments at $(date) ==="

BASE_ARGS="--data-location ./data --balanced-sampler True --model ViT-B-32 --batch-size 32"

# Job 1 (first 3 configurations)
if [ "${1:-1}" = "1" ]; then
    run_experiment "base" "1e-4" "0.0"
    run_experiment "best_single_task" "5e-4" "0.0"
    run_experiment "best_normalized" "1e-4" "0.01"
    
    log "\n=== Cleaning up job 1 files to save space ==="
    # find ./balanced_data_results/base ./balanced_data_results/best_single_task ./balanced_data_results/best_normalized -name "*.pt" ! -name "pretrained.pt" ! -name "*_finetuned.pt" -delete
fi

# Job 2 (remaining 3 configurations)
if [ "${1:-1}" = "2" ]; then
    run_experiment "best_absolute" "1e-5" "0.0"
    run_experiment "best_log_trace" "1e-5" "0.0"
    run_experiment "log_trace_stopping" "1e-4" "0.0" "finetune_log_tr_based.py"
    
    # log "\n=== Cleaning up job 2 files to save space ==="
    # find ./balanced_data_results/best_absolute ./balanced_data_results/best_log_trace ./balanced_data_results/log_trace_stopping -name "*.pt" ! -name "pretrained.pt" ! -name "*_finetuned.pt" -delete
fi

log "\n=== Experiments completed at $(date) ==="
