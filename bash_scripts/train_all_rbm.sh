#!/bin/bash

for DATA in mnist kmnist emnist omniglot caltech
do 
  python rbm_learning.py --sampler cyc_dmala --use_manual_EE --burnin_adaptive --adapt_every 100 --data $DATA --cuda_id $1 --n_hidden 1000 --a_s_cut .5 --num_cycles 1000;
  python rbm_learning.py --sampler cyc_dula --use_manual_EE --burnin_adaptive --adapt_every 100 --data $DATA --cuda_id $1 --n_hidden 1000 --a_s_cut .9 --num_cycles 1000;
  python rbm_learning.py --sampler dmala --step_size .2 --initial_balancing_constant .5 --data $DATA --cuda_id $1 --n_hidden 1000;
done

