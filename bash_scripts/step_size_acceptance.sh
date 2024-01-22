#!/bin/bash

for DATA in mnist kmnist emnist omniglot caltech
do
  ALPHA=.1 ;
  while [$ALPHA -le 20]
  do
    for BAL_CONSTANT in .5 .6 .7 .8 .9 1.0
    do 
      python rbm_sample.py --step_size $STEP_SIZE --samplers dmala --data $DATA \
        --initial_balancing_constant $BAL_CONSTANT --rbm_train_iter 1000 --cuda_id $1 \
        --n_steps 1000
    done
  done
  $ALPHA=$ALPHA+1;
done



