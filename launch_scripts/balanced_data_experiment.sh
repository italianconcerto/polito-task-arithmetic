COMMON_ARGS="--data-location ./data \
    --save ./balanced_data_results \
    --model ViT-B-32 \
    --batch-size 32 \
    --lr 1e-4 \
    --wd 0.0"
    
python finetune.py $COMMON_ARGS" --balanced-sampler True
