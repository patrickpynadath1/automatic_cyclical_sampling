#!/bin/bash
budget=200;
a_s=.5;
seed=1234567;
save_dir=raw_exp_data/rbm_sample;
lr=.5;
sample_steps=5000;
rbm_train_iter=1000;
for data in mnist kmnist emnist
do
  python rbm_mode_escape.py \
    --sampler asb \
    --data $data \
    --cuda_id $1 \
    --n_hidden 500 \
    --rbm_train_iter $rbm_train_iter \
    --n_steps $sample_steps \
    --save_dir $save_dir ;

  python rbm_mode_escape.py \
    --sampler gwg \
    --data $data \
    --cuda_id $1 \
    --n_hidden 500 \
    --rbm_train_iter $rbm_train_iter \
    --n_steps $sample_steps \
    --save_dir $save_dir ;

  python rbm_mode_escape.py \
    --sampler dmala \
    --data $data \
    --cuda_id $1  \
    --n_hidden 500 \
    --rbm_train_iter $rbm_train_iter  \
    --n_steps $sample_steps \
    --initial_balancing_constant .5 \
    --step_size .2 \
    --save_dir $save_dir ;

  python rbm_mode_escape.py \
    --sampler acs \
    --burnin_adaptive \
    --burnin_budget $budget \
    --data $data \
    --cuda_id $1  \
    --a_s_cut $a_s \
    --n_hidden 500  \
    --bal_resolution 10 \
    --rbm_train_iter $rbm_train_iter  \
    --adapt_strat greedy \
    --num_cycles 250\
    --pair_optim \
    --burnin_test_steps 1 \
    --burnin_lr $lr \
    --n_steps $sample_steps \
    --seed $seed  \
    --save_dir $save_dir ;
done