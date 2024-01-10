#!/bin/bash
BUDGET=400;
A_S=.6;
for DATA in mnist kmnist emnist omniglot caltech
do

  python rbm_sample.py \
    --sampler dmala \
    --data $DATA \
    --cuda_id $1  \
    --n_hidden 1000  \
    --rbm_train_iter 1  \
    --n_steps 5000 \
    --initial_balancing_constant .5 \
    --step_size .2; 
  python rbm_sample.py \
    --sampler cyc_dmala \
    --burnin_adaptive \
    --burnin_budget $BUDGET \
    --a_s_cut $A_S \
    --data $DATA \
    --cuda_id $1  \
    --a_s_cut $A_S \
    --n_hidden 1000  \
    --bal_resolution 10 \
    --rbm_train_iter 1  \
    --adapt_strat greedy \
    --num_cycles 500 \
    --pair_optim \
    --burnin_test_steps 1 \
    --n_steps 5000; 
done

