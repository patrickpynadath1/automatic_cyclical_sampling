#!/bin/bash

for TRAIN_ITR in 1 20 
do 
  python rbm_sample.py --samplers dmala dula --use_big --rbm_train_iter $TRAIN_ITR --cuda_id $1

  for STEP_SIZE in .1 .15 .2 .25 .3 .35 .4 .45 .5 .6 .7 .8  
  do
    for BAL_CONSTANT in .5  
    do 
      python rbm_sample.py --step_size $STEP_SIZE --samplers dmala dula \
        --initial_balancing_constant $BAL_CONSTANT --rbm_train_iter $TRAIN_ITR --cuda_id $1
    done
  done

  for STEP_SIZE in 2.0 4.0 5.0 6.0 7.0 8.0 10.0 20.0 
  do
    for BAL_CONSTANT in .95
    do 
      python rbm_sample.py --step_size $STEP_SIZE --samplers dmala dula \
        --initial_balancing_constant $BAL_CONSTANT --rbm_train_iter $TRAIN_ITR --cuda_id $1
    done
  done
done



