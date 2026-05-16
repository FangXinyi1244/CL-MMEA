#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"

CUDA_VISIBLE_DEVICES=$1 python3 src/run.py \
    --file_dir data/DBP15K/$3 \
    --rate 0.3 \
    --lr .0005 \
    --epochs 1000 \
    --hidden_units "300,300,300" \
    --check_point 50  \
    --bsize 512 \
    --il \
    --il_start 500 \
    --semi_learn_step 5 \
    --csls \
    --csls_k 3 \
    --seed $2 \
    --tau 0.1 \
    --tau2 4.0 \
    --structure_encoder "gat" \
    --img_dim 100 \
    --attr_dim 100 \
    --name_dim 100 \
    --char_dim 100 \
    --mask_ratio 0.15 \
    --mask_method "random" \
    --mask_loss_weight 0.1 \
    --use_hard_negatives \
    --hard_negative_k 50
