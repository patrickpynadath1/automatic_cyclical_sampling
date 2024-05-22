#!/bin/bash
budget=200;
a_s=.5;
seed=1234567;
lr=.5;
save_dir=raw_exp_data/rbm_sample;
sample_steps=1;
rbm_train_iter=1000;
n_steps=10000;
for data in mnist kmnist emnist omniglot caltech 
do
  for a_s in .3 .4 .5 .6 .7 .8 .9
  do
    python rbm_sample.py \
      --sampler acs \
      --burnin_budget $budget \
      --data $data \
      --cuda_id $1  \
      --a_s_cut $a_s \
      --n_hidden 500  \
      --bal_resolution 10 \
      --rbm_train_iter 1000  \
      --num_cycles 500\
      --burnin_lr $lr \
      --n_steps $n_steps  \
      --rbm_train_iter $rbm_train_iter\
      --save_dir ${save_dir}_a_s_${a_s};
  done


  # for num_c in 50 100 250 500 1000 5000
  # do
  #   python rbm_sample.py \
  #     --sampler acs \
  #     --burnin_budget $budget \
  #     --data $data \
  #     --cuda_id $1  \
  #     --a_s_cut .5 \
  #     --n_hidden 500  \
  #     --bal_resolution 10 \
  #     --rbm_train_iter 1000  \
  #     --num_cycles $num_c\
  #     --burnin_lr $lr \
  #     --n_steps $n_steps  \
  #     --rbm_train_iter $rbm_train_iter\
  #     --save_dir ${save_dir}_numc_${num_c};
  # done

  for b_max in .6 .7 .8 .9 1.0
  do
    python rbm_sample.py \
      --sampler acs \
      --burnin_budget $budget \
      --data $data \
      --cuda_id $1  \
      --a_s_cut .5 \
      --n_hidden 500  \
      --bal_resolution 10 \
      --rbm_train_iter 1000  \
      --num_cycles 500 \
      --burnin_lr $lr \
      --burnin_big_bal $b_max \
      --n_steps $n_steps  \
      --rbm_train_iter $rbm_train_iter\
      --save_dir ${save_dir}_bmax_${b_max};
  done
done 