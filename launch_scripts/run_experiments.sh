#!/bin/bash
set -e  # Exit on error

# Base directory for results
BASE_DIR="./results"

# Create necessary directories
mkdir -p "$BASE_DIR"

# 1. Complexity & batch size
echo "=== Running batch size experiments ==="
for batch_size in 8 16 64 128; do
    exp_dir="$BASE_DIR/batch_size_${batch_size}"
    mkdir -p "$exp_dir"
    
    echo "Running experiment with batch_size=$batch_size"
    python experiment.py \
        --data-location ./data \
        --save "$exp_dir" \
        --batch-size "$batch_size" \
        --lr 1e-4 \
        --wd 0.0
done

# 2. Complexity & learning rate
echo "=== Running learning rate experiments ==="
for lr in 5e-4 5e-5 1e-5; do
    exp_dir="$BASE_DIR/learning_rate_${lr}"
    mkdir -p "$exp_dir"
    
    echo "Running experiment with lr=$lr"
    python experiment.py \
        --data-location ./data \
        --save "$exp_dir" \
        --batch-size 32 \
        --lr "$lr" \
        --wd 0.0
done

# 3. Complexity & weight decay
echo "=== Running weight decay experiments ==="
for wd in 0.001 0.01 0.1; do
    exp_dir="$BASE_DIR/weight_decay_${wd}"
    mkdir -p "$exp_dir"
    
    echo "Running experiment with wd=$wd"
    python experiment.py \
        --data-location ./data \
        --save "$exp_dir" \
        --batch-size 32 \
        --lr 1e-4 \
        --wd "$wd"
done

# 4. Complexity & stopping criteria
# Note: This is handled by the existing code in finetune.py and eval_single_task.py
# which already saves and evaluates both best FIM and best accuracy models
echo "=== Running stopping criteria experiment ==="
exp_dir="$BASE_DIR/stopping_criteria"
mkdir -p "$exp_dir"

python experiment.py \
    --data-location ./data \
    --save "$exp_dir" \
    --batch-size 32 \
    --lr 1e-4 \
    --wd 0.0

# Combine all results into a single spreadsheet
echo "=== Combining all results ==="
python combine_results.py --results-dir "$BASE_DIR" 