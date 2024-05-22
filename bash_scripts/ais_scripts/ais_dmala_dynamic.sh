#!/bin/bash
data=dynamic_mnist;
sampler=dmala;
python eval_ais.py \
  --dataset_name $data \
  --sampler gwg \
  --step_size 0.2 \
  --sampling_steps 40 \
  --model resnet-64 \
  --buffer_size 10000 \
  --cuda_id $1 \
  --n_iters 300000 \
  --base_dist \
  --n_samples 500 \
  --eval_sampling_steps 300000 \
  --ema \
  --viz_every 1000 \
  --save_dir figs/ebm/${sampler}_0.15_ss10/$data \
  --ckpt_path best_ckpt_${data}_dmala_0.15.pt;
