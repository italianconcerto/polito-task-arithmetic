# Create a directory for saving results

# 1. Fine-tune on all datasets
python finetune.py \
    --data-location ./data \
    --save ./results \
    --batch-size 32 \
    --lr 1e-4 \
    --wd 0.0

# 2. Evaluate single-task performance
python eval_single_task.py \
    --data-location ./data \
    --save ./results

# 3. Perform task addition
python eval_task_addition.py \
    --data-location ./data \
    --save ./results