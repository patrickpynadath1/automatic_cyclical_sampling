#!/bin/bash

for DATA in mnist kmnist emnist omniglot caltech
do
 python rbm_learning.py \
   --sampler gwg \
   --data $DATA \
   --sampling_steps 10 \
   --cuda_id $1 \
   --n_hidden 500 \
   --save_dir figs/rbm_learn_ss_10;
 python rbm_learning.py --sampler dmala \
   --step_size .2 \
   --sampling_steps 10 \
   --initial_balancing_constant .5 --data $DATA --cuda_id $1 \
   --n_hidden 500 \
   --save_dir figs/rbm_learn_ss_10;
done

python rbm_learning.py \
  --sampler acs \
  --use_manual_EE \
  --adapt_every 50 \
  --data mnist \
  --sampling_steps 10 \
  --cuda_id $1 \
  --n_hidden 500 \
  --a_s_cut .5 \
  --num_cycles 250 \
  --burnin_budget 200 \
  --burnin_lr .5 \
  --big_step .5 \
  --burnin_big_bal .9 \
  --save_dir figs/rbm_learn_ss10;

python rbm_learning.py \
  --sampler acs \
  --use_manual_EE \
  --adapt_every 50 \
  --data caltech \
  --cuda_id $1 \
  --sampling_steps 10 \
  --n_hidden 500 \
  --a_s_cut .5 \
  --num_cycles 100 \
  --burnin_budget 50 \
  --burnin_lr .5 \
  --big_step .5 \
  --burnin_big_bal .9 \
  --save_dir figs/rbm_learn_ss10;


python rbm_learning.py \
  --sampler acs \
  --use_manual_EE \
  --adapt_every 50\
  --data omniglot \
  --cuda_id $1 \
  --sampling_steps 10 \
  --n_hidden 500 \
  --a_s_cut .5 \
  --num_cycles 100 \
  --burnin_budget 200 \
  --burnin_lr .5 \
  --big_step .5 \
  --burnin_big_bal .9 \
  --save_dir figs/rbm_learn_ss10;


python rbm_learning.py \
  --sampler acs \
  --use_manual_EE \
  --adapt_every 50\
  --data emnist \
  --cuda_id $1 \
  --n_hidden 500 \
  --a_s_cut .5 \
  --sampling_steps 10 \
  --num_cycles 250 \
  --burnin_budget 200 \
  --burnin_lr .5 \
  --big_step .5 \
  --burnin_big_bal .9 \
  --save_dir figs/rbm_learn_ss10;


python rbm_learning.py \
  --sampler acs \
  --use_manual_EE \
  --adapt_every 50\
  --data kmnist \
  --cuda_id $1 \
  --n_hidden 500 \
  --sampling_steps 10 \
  --a_s_cut .5 \
  --num_cycles 250 \
  --burnin_budget 200 \
  --burnin_lr .5 \
  --big_step .5 \
  --burnin_big_bal .9 \
  --save_dir figs/rbm_learn_ss10;