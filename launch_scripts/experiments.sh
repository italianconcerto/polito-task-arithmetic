# Create directories for different experiments
mkdir -p experiments/{batch_size,learning_rate,weight_decay}

# 1. Batch Size Experiments
for bs in 8 16 32 64 128; do
    echo "Running batch size: $bs"
    python finetune.py \
        --data-location ./data \
        --save ./experiments/batch_size/bs_${bs} \
        --batch-size $bs \
        --lr 1e-4 \
        --wd 0.0
    
    python eval_single_task.py \
        --data-location ./data \
        --save ./experiments/batch_size/bs_${bs}
    
    python eval_task_addition.py \
        --data-location ./data \
        --save ./experiments/batch_size/bs_${bs}
done

# 2. Learning Rate Experiments
for lr in 5e-4 5e-5 1e-5; do
    echo "Running learning rate: $lr"
    python finetune.py \
        --data-location ./data \
        --save ./experiments/learning_rate/lr_${lr} \
        --batch-size 32 \
        --lr $lr \
        --wd 0.0
    
    python eval_single_task.py \
        --data-location ./data \
        --save ./experiments/learning_rate/lr_${lr}
    
    python eval_task_addition.py \
        --data-location ./data \
        --save ./experiments/learning_rate/lr_${lr}
done

# 3. Weight Decay Experiments
for wd in 0.001 0.01 0.1; do
    echo "Running weight decay: $wd"
    python finetune.py \
        --data-location ./data \
        --save ./experiments/weight_decay/wd_${wd} \
        --batch-size 32 \
        --lr 1e-4 \
        --wd $wd
    
    python eval_single_task.py \
        --data-location ./data \
        --save ./experiments/weight_decay/wd_${wd}
    
    python eval_task_addition.py \
        --data-location ./data \
        --save ./experiments/weight_decay/wd_${wd}
done