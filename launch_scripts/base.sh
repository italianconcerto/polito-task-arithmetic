#!/bin/bash
set -e  # Exit on error

# Create necessary directories
mkdir -p results
mkdir -p data

# 0. Initialize and save pre-trained model
echo "Initializing pre-trained model..."
python init_pretrained.py \
    --save ./results

# 1. Fine-tune on all datasets
echo "Fine-tuning on all datasets..."
python finetune.py \
    --data-location ./data \
    --save ./results \
    --batch-size 32 \
    --lr 1e-4 \
    --wd 0.0

# 2. Evaluate single-task performance
echo "Evaluating single-task performance..."
python eval_single_task.py \
    --data-location ./data \
    --save ./results

# 3. Perform task addition
echo "Performing task addition..."
python eval_task_addition.py \
    --data-location ./data \
    --save ./results