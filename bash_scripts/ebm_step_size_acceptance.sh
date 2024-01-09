#!/bin/bash
#

python ebm_sample.py --sampler dmala --base_dist --use_big --cuda_id $1

for STEP_SIZE in .1 .15 .2 .25 .3 .35 .4 .45 .5 .6 .7 .8  
do
  for BAL_CONSTANT in .5  
  do 
    python ebm_sample.py --step_size $STEP_SIZE --sampler dmala \
      --initial_balancing_constant $BAL_CONSTANT --base_dist --cuda_id $1
  done
done

for STEP_SIZE in .2 .3 .4 .5 .6 .7 .8 .9 1.0 1.5 2.0 4.0 
do
  for BAL_CONSTANT in .95
  do 
    python ebm_sample.py --step_size $STEP_SIZE --sampler dmala \
      --initial_balancing_constant $BAL_CONSTANT --base_dist --cuda_id $1
  done
done

