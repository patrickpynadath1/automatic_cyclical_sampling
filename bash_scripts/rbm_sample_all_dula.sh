#!/bin/bash
BUDGET=1000;
A_S=.975;
LR=.5;
N_STEPS=5000;
SEED=1234567;
for DATA in mnist kmnist emnist omniglot caltech
do
  # python rbm_sample.py \
  #   --sampler dula \
  #   --data $DATA \
  #   --n_hidden 500  \
  #   --cuda_id $1 \
  #   --rbm_train_iter 1000 \
  #   --cd 100 \
  #   --step_size .1 \
  #   --seed $SEED \
  #   --n_steps $N_STEPS \
  #   --initial_balancing_constant .5 ;
  python rbm_sample.py \
    --sampler cyc_dula \
    --burnin_adaptive \
    --burnin_budget $BUDGET \
    --a_s_cut $A_S \
    --data $DATA \
    --cuda_id $1  \
    --pair_optim \
    --n_hidden 500  \
    --bal_resolution 10 \
    --rbm_train_iter 1000 \
    --cd 100 \
    --seed $SEED \
    --adapt_strat greedy \
    --num_cycles 500 \
    --n_steps $N_STEPS \
    --burnin_test_steps 1 \
    --burnin_lr $LR;

done

