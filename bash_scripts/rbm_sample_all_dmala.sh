#!/bin/bash
BUDGET=200;
A_S=.5;
SEED=1234567;
LR=.2;
for DATA in mnist kmnist emnist omniglot caltech
do

  # python rbm_sample.py \
  #   --sampler dmala \
  #   --data $DATA \
  #   --cuda_id $1  \
  #   --n_hidden 500 \
  #   --rbm_train_iter 1000  \
  #   --n_steps 5000 \
  #   --initial_balancing_constant 1.0 \
  #   --step_size 10.0; 
  python rbm_sample.py \
    --sampler dmala \
    --burnin_adaptive \
    --a_s_cut $A_S \
    --burnin_budget $BUDGET \ 
    --data $DATA \
    --cuda_id $1  \
    --n_hidden 500 \
    --rbm_train_iter 1000  \
    --n_steps 5000 \
    --initial_balancing_constant .5 \
    --step_size .2; 
  python rbm_sample.py \
    --sampler dmala \
    --data $DATA \
    --cuda_id $1  \
    --n_hidden 500 \
    --rbm_train_iter 1000  \
    --n_steps 5000 \
    --initial_balancing_constant .5 \
    --step_size .2; 
  python rbm_sample.py \
    --sampler cyc_dmala \
    --burnin_adaptive \
    --burnin_budget $BUDGET \
    --data $DATA \
    --cuda_id $1  \
    --a_s_cut $A_S \
    --n_hidden 500  \
    --bal_resolution 10 \
    --rbm_train_iter 1000  \
    --adapt_strat greedy \
    --num_cycles 250 \
    --use_manual_EE \
    --pair_optim \
    --burnin_test_steps 1 \
    --burnin_lr $LR \
    --n_steps 5000 \
    --seed $SEED ;
done

