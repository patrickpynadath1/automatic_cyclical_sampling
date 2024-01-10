#!/bin/bash
BUDGET=400;
A_S=.6;
LR=.5;
SEED=420420;
for DATA in mnist kmnist emnist omniglot caltech
do
  # python rbm_sample.py \
  #   --sampler dula \
  #   --data $DATA \
  #   --cuda_id $1 \
  #   --rbm_train_iter 1 \
  #   --step_size .1 \
  #   --seed $SEED \
  #   --n_steps 5000 \
  #   --initial_balancing_constant .5 ;
  # python rbm_sample.py \
  #   --sampler cyc_dula \
  #   --burnin_adaptive \
  #   --burnin_budget $BUDGET \
  #   --a_s_cut $A_S \
  #   --data $DATA \
  #   --cuda_id $1  \
  #   --a_s_cut $A_S \
  #   --pair_optim \
  #   --n_hidden 1000  \
  #   --bal_resolution 10 \
  #   --rbm_train_iter 1  \
  #   --adapt_strat greedy \
  #   --num_cycles 500 \
  #   --n_steps 5000 \
  #   --burnin_test_steps 1 \
  #   --burnin_lr $LR \
  #   --seed $SEED;


  python rbm_sample.py \
    --sampler cyc_dula \
    --burnin_adaptive \
    --burnin_budget 50 \
    --a_s_cut $A_S \
    --data $DATA \
    --cuda_id $1  \
    --a_s_cut $A_S \
    --n_hidden 1000  \
    --bal_resolution 10 \
    --rbm_train_iter 1  \
    --adapt_strat bayes \
    --num_cycles 500 \
    --n_steps 5000 \
    --burnin_test_steps 1 \
    --burnin_lr $LR \
    --seed $SEED;
done

