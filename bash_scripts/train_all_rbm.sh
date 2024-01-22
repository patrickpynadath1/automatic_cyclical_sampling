#!/bin/bash

for DATA in mnist kmnist emnist omniglot caltech
do 
  python rbm_learning.py \
    --sampler cyc_dmala \
    --use_manual_EE \
    --adapt_every 10\
    --data $DATA \
    --cuda_id $1 \
    --n_hidden 500 \
    --a_s_cut .6 \
    --num_cycles 50 \
    --burnin_budget 200 \
    --burnin_adaptive \
    --burnin_lr .5 \
    --big_step .5;
  # python rbm_learning.py \
  #   --sampler cyc_dmala \
  #   --use_manual_EE \
  #   --adapt_every 10\
  #   --data $DATA \
  #   --cuda_id $1 \
  #   --n_hidden 500 \
  #   --a_s_cut .5 \
  #   --num_cycles 500 \
  #   --burnin_budget 50 \
  #   --burnin_lr .1 \
  #   --big_step .75;
  # python rbm_learning.py \
  #   --sampler cyc_dmala \
  #   --use_manual_EE \
  #   --adapt_every 10\
  #   --data $DATA \
  #   --cuda_id $1 \
  #   --n_hidden 500 \
  #   --a_s_cut .5 \
  #   --num_cycles 500 \
  #   --burnin_budget 50 \
  #   --burnin_lr .1 \
  #   --big_step 2.0;
  # python rbm_learning.py \
  #   --sampler cyc_dmala \
  #   --use_manual_EE \
  #   --adapt_every 10\
  #   --data $DATA \
  #   --cuda_id $1 \
  #   --n_hidden 500 \
  #   --a_s_cut .5 \
  #   --num_cycles 500 \
  #   --burnin_budget 50 \
  #   --burnin_lr .1 \
  #   --big_step 1.0;
  # python rbm_learning.py --sampler dmala --step_size .2 --initial_balancing_constant .5 --data $DATA --cuda_id $1 --n_hidden 500;
  # python rbm_learning.py --sampler gb --data $DATA --cuda_id $1 --n_hidden 500;
done

