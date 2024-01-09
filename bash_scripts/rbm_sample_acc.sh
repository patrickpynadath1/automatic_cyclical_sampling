#!/bin/bash

for DATA in kmnist emnist omniglot caltech mnist
do 

  for STEP_SIZE in .5 .75 1.0 1.5 2.0 2.5 3.0 3.5 4.0 
  do
    for BAL_CONSTANT in .75 .8 .85 .9 .95
    do 
      python rbm_sample.py --step_size $STEP_SIZE --samplers dmala --rbm_lr .001\
        --initial_balancing_constant $BAL_CONSTANT --rbm_train_iter 10000 \
        --cuda_id $1 --data $DATA --use_dula_init --n_hidden 1000 --use_dmala_trained_rbm
    done
  done
done
