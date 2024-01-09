#!/bin/bash
BUDGET=100;
A_S=.5;
for DATA in mnist kmnist emnist omniglot caltech
do
  python rbm_sample.py \
    --sampler dula \
    --data $DATA \
    --cuda_id $1 \
    --rbm_train_iter 10000 \
    --step_size .1 \
    --initial_balancing_constant .5 ;
  python rbm_sample.py \
    --sampler dula \
    --data $DATA \
    --cuda_id $1 \
    --rbm_train_iter 10000 \
    --use_dmala_trained_rbm \
    --step_size .1 \
    --initial_balancing_constant .5 ;
  python rbm_sample.py \
    --sampler dula \
    --data $DATA \
    --cuda_id $1 \
    --rbm_train_iter 1 \
    --step_size .1 \
    --initial_balancing_constant .5 ;
  python rbm_sample.py \
    --sampler dula \
    --burnin_adaptive \
    --burnin_budget $BUDGET \
    --data $DATA \
    --cuda_id $1 \
    --a_s_cut $A_S \
    --rbm_train_iter 10000 \
    --burnin_test_steps 1 \
    --step_obj alpha_min;
  python rbm_sample.py \
    --sampler dula \
    --burnin_adaptive \
    --burnin_budget $BUDGET \
    --data $DATA \
    --a_s_cut $A_S \
    --cuda_id $1 \
    --rbm_train_iter 10000 \
    --use_dmala_trained_rbm \
    --burnin_test_steps 1 \
    --step_obj alpha_min;
  python rbm_sample.py \
    --sampler dula \
    --burnin_adaptive \
    --a_s_cut $A_S \
    --burnin_budget $BUDGET \
    --data $DATA \
    --cuda_id $1 \
    --rbm_train_iter 1 \
    --burnin_test_steps 1 \
    --step_obj alpha_min;
  python rbm_sample.py \
    --sampler dula \
    --burnin_adaptive \
    --burnin_budget $BUDGET \
    --data $DATA \
    --cuda_id $1 \
    --a_s_cut $A_S \
    --rbm_train_iter 10000 \
    --burnin_test_steps 1 \
    --step_obj alpha_max;
  python rbm_sample.py \
    --sampler dula \
    --burnin_adaptive \
    --burnin_budget $BUDGET \
    --data $DATA \
    --a_s_cut $A_S \
    --cuda_id $1 \
    --rbm_train_iter 10000 \
    --use_dmala_trained_rbm \
    --burnin_test_steps 1 \
    --step_obj alpha_max;
  python rbm_sample.py \
    --sampler dula \
    --burnin_adaptive \
    --a_s_cut $A_S \
    --burnin_budget $BUDGET \
    --data $DATA \
    --cuda_id $1 \
    --rbm_train_iter 1 \
    --burnin_test_steps 1 \
    --step_obj alpha_max;
  python rbm_sample.py \
    --sampler cyc_dula \
    --burnin_adaptive \
    --burnin_budget $BUDGET \
    --a_s_cut $A_S \
    --data $DATA \
    --cuda_id $1  \
    --a_s_cut .5\
    --n_hidden 1000  \
    --bal_resolution 5 \
    --rbm_train_iter 10000 \
    --use_dmala_trained_rbm \
    --adapt_strat bayes \
    --num_cycles 500 \
    --burnin_test_steps 1; 
  python rbm_sample.py \
    --sampler cyc_dula \
    --burnin_adaptive \
    --burnin_budget $BUDGET \
    --data $DATA \
    --cuda_id $1  \
    --a_s_cut $A_S \
    --a_s_cut .5 \
    --n_hidden 1000  \
    --bal_resolution 5 \
    --rbm_train_iter 10000  \
    --adapt_strat bayes \
    --num_cycles 500 \
    --burnin_test_steps 1; 
  python rbm_sample.py \
    --sampler cyc_dula \
    --burnin_adaptive \
    --burnin_budget $BUDGET \
    --a_s_cut $A_S \
    --data $DATA \
    --cuda_id $1  \
    --a_s_cut .5 \
    --n_hidden 1000  \
    --bal_resolution 5 \
    --rbm_train_iter 1  \
    --adapt_strat bayes \
    --num_cycles 500 \
    --burnin_test_steps 1; 
done

