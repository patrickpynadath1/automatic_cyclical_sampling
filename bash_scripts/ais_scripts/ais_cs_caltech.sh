#!/bin/bash
data=caltech;

python eval_ais.py \
  --dataset_name $data \
  --sampler gwg \
  --cuda_id $1 \
  --step_size 0.2 \
  --sampling_steps 40 \
  --model resnet-64 \
  --buffer_size 1000 \
  --n_iters 300000 \
  --base_dist \
  --n_samples 500 \
  --eval_sampling_steps 300000 \
  --ema \
  --viz_every 1000 \
  --save_dir figs/ebm/cs/${data}_0.8_20 \
  --ckpt_path best_ckpt_${data}_cyc_dmala_1.5.pt;
