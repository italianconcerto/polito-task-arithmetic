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

log "=== Starting balanced data experiments at $(date) ==="

BASE_ARGS="--data-location ./data --balanced-sampler True --model ViT-B-32 --batch-size 32"

log "\n=== Base configuration ==="
run_cmd "python finetune.py $BASE_ARGS --save ./balanced_data_results/base --lr 1e-4 --wd 0.0"
run_cmd "python eval_single_task.py $BASE_ARGS --save ./balanced_data_results/base --lr 1e-4 --wd 0.0"
run_cmd "python eval_task_addition.py $BASE_ARGS --save ./balanced_data_results/base --lr 1e-4 --wd 0.0"

log "\n=== Best single task accuracy configuration ==="
run_cmd "python finetune.py $BASE_ARGS --save ./balanced_data_results/best_single_task --lr 5e-4 --wd 0.0"
run_cmd "python eval_single_task.py $BASE_ARGS --save ./balanced_data_results/best_single_task --lr 5e-4 --wd 0.0"
run_cmd "python eval_task_addition.py $BASE_ARGS --save ./balanced_data_results/best_single_task --lr 5e-4 --wd 0.0"

log "\n=== Best normalized accuracy configuration ==="
run_cmd "python finetune.py $BASE_ARGS --save ./balanced_data_results/best_normalized --lr 1e-4 --wd 0.01"
run_cmd "python eval_single_task.py $BASE_ARGS --save ./balanced_data_results/best_normalized --lr 1e-4 --wd 0.01"
run_cmd "python eval_task_addition.py $BASE_ARGS --save ./balanced_data_results/best_normalized --lr 1e-4 --wd 0.01"

log "\n=== Best absolute accuracy configuration ==="
run_cmd "python finetune.py $BASE_ARGS --save ./balanced_data_results/best_absolute --lr 1e-5 --wd 0.0"
run_cmd "python eval_single_task.py $BASE_ARGS --save ./balanced_data_results/best_absolute --lr 1e-5 --wd 0.0"
run_cmd "python eval_task_addition.py $BASE_ARGS --save ./balanced_data_results/best_absolute --lr 1e-5 --wd 0.0"

log "\n=== Best log-trace configuration ==="
run_cmd "python finetune.py $BASE_ARGS --save ./balanced_data_results/best_log_trace --lr 1e-5 --wd 0.0"
run_cmd "python eval_single_task.py $BASE_ARGS --save ./balanced_data_results/best_log_trace --lr 1e-5 --wd 0.0"
run_cmd "python eval_task_addition.py $BASE_ARGS --save ./balanced_data_results/best_log_trace --lr 1e-5 --wd 0.0"

log "\n=== Log-trace stopping criteria configuration ==="
run_cmd "python finetune_log_tr_based.py $BASE_ARGS --save ./balanced_data_results/log_trace_stopping --lr 1e-4 --wd 0.0"
run_cmd "python eval_single_task.py $BASE_ARGS --save ./balanced_data_results/log_trace_stopping --lr 1e-4 --wd 0.0"
run_cmd "python eval_task_addition.py $BASE_ARGS --save ./balanced_data_results/log_trace_stopping --lr 1e-4 --wd 0.0"

log "\n=== Experiments completed at $(date) ==="
