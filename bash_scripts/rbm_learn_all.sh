#!/bin/bash

for DATA in mnist kmnist emnist omniglot caltech
do
#  python rbm_learning.py \
#    --sampler gwg \
#    --data $DATA \
#    --cuda_id $1 \
#    --n_hidden 500 ;
  python rbm_learning.py \
    --sampler acs \
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
#  python rbm_learning.py --sampler dmala --step_size .2 --initial_balancing_constant .5 --data $DATA --cuda_id $1 --n_hidden 500;
#  python rbm_learning.py --sampler gb --data $DATA --cuda_id $1 --n_hidden 500;
done

