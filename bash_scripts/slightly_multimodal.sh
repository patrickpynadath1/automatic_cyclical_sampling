#!/bin/bash

BATCH_SIZE=128;
NUM_MODES=5;
SPACE_BETWEEN_MODES=75;
VAR=.2;


python multi_modal_ex.py \
 --sampler acs \
 --step_size .007\
 --min_lr .003 \
 --initial_balancing_constant .5 \
 --num_modes $NUM_MODES\
 --space_between_modes $SPACE_BETWEEN_MODES \
 --dist_var $VAR \
 --sampling_steps 10000 \
 --num_cycles 1000 \
 --batch_size $BATCH_SIZE \
 --dist_weights 1.0 0.4 0.4 0.4 \
 --slightly_multi_modal; 

python multi_modal_ex.py \
  --sampler dmala \
  --step_size .007\
  --initial_balancing_constant .5 \
  --num_modes $NUM_MODES\
  --space_between_modes $SPACE_BETWEEN_MODES \
  --dist_var $VAR \
  --sampling_steps 10000 \
  --num_cycles 500 \
  --batch_size $BATCH_SIZE \
  --slightly_multi_modal; 