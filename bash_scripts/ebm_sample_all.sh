#!/bin/bash
budget=200;
a_s=.8;
seed=1234567;
lr=.5;
sampling_steps=5000;
for data in caltech
do 
#   python ebm_sample.py \
#     --sampler dmala \
#     --cuda_id $1  \
#     --sampling_steps $sampling_steps \
#     --initial_balancing_constant .5 \
#     --base_dist \
#     --step_size .2 \
#     --dataset_name $data \
#     --use_acs_ebm;
  python ebm_sample.py \
    --sampler acs \
    --burnin_budget $budget \
    --dataset_name $data \
    --cuda_id $1  \
    --a_s_cut $a_s \
    --bal_resolution 10 \
    --num_cycles 50 \
    --burnin_big_bal .9 \
    --burnin_lr $lr \
    --base_dist \
    --sampling_steps $sampling_steps \
    --seed $seed;
  #  python ebm_sample.py \
  #    --sampler gwg \
  #    --dataset_name $data \
  #    --cuda_id $1 \
  #    --base_dist \
  #    --sampling_steps $sampling_steps;
  #  python ebm_sample.py \
  #    --sampler asb\
  #    --dataset_name $data \
  #    --cuda_id $1 \
  #    --base_dist \
  #    --sampling_steps $sampling_steps;
done
