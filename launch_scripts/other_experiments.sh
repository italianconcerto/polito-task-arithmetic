#!/bin/bash

# Exit on any error
set -e

# Parse command line arguments
skip_init=false
skip_finetune=false
only_task_addition=false

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
        --only-task-addition)
            only_task_addition=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--skip-init] [--skip-finetune] [--only-task-addition]"
            exit 1
            ;;
    esac
done

# Create necessary directories
mkdir -p data;
mkdir -p logs;
mkdir -p other_results/batch_size;
mkdir -p other_results/learning_rate;
mkdir -p other_results/weight_decay;
mkdir -p other_results/stopping_criteria;


# Get current timestamp for log file
timestamp=$(date +"%Y%m%d_%H%M%S")
log_file="logs/other_experiments_${timestamp}.log"

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
for batch_size in 8 16 64 128
do
    # Common arguments used across commands
    COMMON_ARGS="--data-location ./data \
        --save ./other_results/batch_size/$batch_size \
        --model ViT-B-32 \
        --batch-size $batch_size \
        --lr 1e-4 \
        --wd 0.0"

    mkdir -p other_results/batch_size/$batch_size
    
    if [ "$only_task_addition" = true ]; then
        # Run only task addition evaluation
        log "\n=== Performing task addition ==="
        run_cmd "python eval_task_addition.py $COMMON_ARGS"
    else
        # Run full experiment pipeline
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
        find ./other_results/batch_size/$batch_size -name "*.pt" ! -name "pretrained.pt" ! -name "*_finetuned.pt" -delete
    fi
done



for learning_rate in 0.0005 0.00005 0.00001
do
    # Common arguments used across commands
    COMMON_ARGS="--data-location ./data \
        --save ./other_results/learning_rate/$learning_rate \
        --model ViT-B-32 \
        --batch-size 32 \
        --lr $learning_rate \
        --wd 0.0"

    mkdir -p other_results/learning_rate/$learning_rate
    
    if [ "$only_task_addition" = true ]; then
        # Run only task addition evaluation
        log "\n=== Performing task addition ==="
        run_cmd "python eval_task_addition.py $COMMON_ARGS"
    else
        # Run full experiment pipeline
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
        find ./other_results/learning_rate/$learning_rate -name "*.pt" ! -name "pretrained.pt" ! -name "*_finetuned.pt" -delete
    fi
done



for weight_decay in 0.001 0.01 0.1
do
    # Common arguments used across commands
    COMMON_ARGS="--data-location ./data \
        --save ./other_results/weight_decay/$weight_decay \
        --model ViT-B-32 \
        --batch-size 32 \
        --lr 1e-4 \
        --wd $weight_decay"

    mkdir -p other_results/weight_decay/$weight_decay
    
    if [ "$only_task_addition" = true ]; then
        # Run only task addition evaluation
        log "\n=== Performing task addition ==="
        run_cmd "python eval_task_addition.py $COMMON_ARGS"
    else
        # Run full experiment pipeline
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
        find ./other_results/weight_decay/$weight_decay -name "*.pt" ! -name "pretrained.pt" ! -name "*_finetuned.pt" -delete
    fi
done


COMMON_ARGS="--data-location ./data \
    --save ./other_results/stopping_criteria/val_acc \
    --model ViT-B-32 \
    --batch-size 32 \
    --lr 1e-4 \
    --wd 0.0"

mkdir -p other_results/stopping_criteria/val_acc

if [ "$only_task_addition" = true ]; then
    # Run only task addition evaluation
    log "\n=== Performing task addition ==="
    run_cmd "python eval_task_addition.py $COMMON_ARGS"
else
    # Run full experiment pipeline
    # 0. Initialize pre-trained model (if not skipped)
    if [ "$skip_init" = false ]; then
        log "\n=== Initializing pre-trained model ==="
        run_cmd "python init_pretrained.py $COMMON_ARGS"
    fi

    # 1. Fine-tune on all datasets (if not skipped)
    if [ "$skip_finetune" = false ]; then
        log "\n=== Fine-tuning on all datasets ==="
        run_cmd "python finetune_acc_based.py $COMMON_ARGS"
    fi

    # 2. Evaluate single-task performance
    log "\n=== Evaluating single-task performance ==="
    run_cmd "python eval_single_task.py $COMMON_ARGS"

    # 3. Perform task addition
    log "\n=== Performing task addition ==="
    run_cmd "python eval_task_addition.py $COMMON_ARGS"

    # Clean up intermediate files to save space
    log "\n=== Cleaning up to save space ==="
    find ./other_results/stopping_criteria/val_acc -name "*.pt" ! -name "pretrained.pt" ! -name "*_finetuned.pt" -delete
fi



COMMON_ARGS="--data-location ./data \
    --save ./other_results/stopping_criteria/fim_logtr \
    --model ViT-B-32 \
    --batch-size 32 \
    --lr 1e-4 \
    --wd 0.0"

mkdir -p other_results/stopping_criteria/fim_logtr

if [ "$only_task_addition" = true ]; then
    # Run only task addition evaluation
    log "\n=== Performing task addition ==="
    run_cmd "python eval_task_addition.py $COMMON_ARGS"
else
    # Run full experiment pipeline
    # 0. Initialize pre-trained model (if not skipped)
    if [ "$skip_init" = false ]; then
        log "\n=== Initializing pre-trained model ==="
        run_cmd "python init_pretrained.py $COMMON_ARGS"
    fi

    # 1. Fine-tune on all datasets (if not skipped)
    if [ "$skip_finetune" = false ]; then
        log "\n=== Fine-tuning on all datasets ==="
        run_cmd "python finetune_log_tr_based.py $COMMON_ARGS"
    fi

    # 2. Evaluate single-task performance
    log "\n=== Evaluating single-task performance ==="
    run_cmd "python eval_single_task.py $COMMON_ARGS"

    # 3. Perform task addition
    log "\n=== Performing task addition ==="
    run_cmd "python eval_task_addition.py $COMMON_ARGS"

    # Clean up intermediate files to save space
    log "\n=== Cleaning up to save space ==="
    find ./other_results/stopping_criteria/fim_logtr -name "*.pt" ! -name "pretrained.pt" ! -name "*_finetuned.pt" -delete
fi

log "\n=== Experiment completed at $(date) ==="

