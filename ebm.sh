#!/bin/bash

python -u pcd_ebm_ema.py \
    --dataset_name dynamic_mnist \
    --sampler dula \
    --step_size 0.1 \
    --sampling_steps 40 \
    --viz_every 100 \
    --model resnet-64 \
    --print_every 10 \
    --lr .0001 \
    --warmup_iters 2000 \
    --buffer_size 2000 \
    --n_iters 10000 \
    --buffer_init mean \
    --base_dist \
    --reinit_freq 0.0 \
    --eval_every 1000 \
    --eval_sampling_steps 2000 ;