#!/bin/bash

BATCH_SIZE=128;
NUM_MODES=5;
SPACE_BETWEEN_MODES=75;
VAR=.15;

python multi_modal_ex.py \
 --sampler acs \
 --step_size .007\
 --min_lr .003 \
 --initial_balancing_constant .5 \
 --num_modes $NUM_MODES\
 --space_between_modes $SPACE_BETWEEN_MODES \
 --dist_var $VAR \
 --sampling_steps 10000 \
 --num_cycles 1 \
 --batch_size $BATCH_SIZE;

python multi_modal_ex.py \
 --sampler rw \
 --step_size .062\
 --initial_balancing_constant .5 \
 --num_modes $NUM_MODES\
 --space_between_modes $SPACE_BETWEEN_MODES \
 --dist_var $VAR \
 --sampling_steps 1000 \
 --batch_size $BATCH_SIZE;

python multi_modal_ex.py \
 --sampler asb \
 --step_size .006\
 --num_modes $NUM_MODES\
 --space_between_modes $SPACE_BETWEEN_MODES \
 --dist_var $VAR \
 --sampling_steps 1000 \
 --batch_size $BATCH_SIZE;

python multi_modal_ex.py \
  --sampler dmala \
  --step_size .011\
  --initial_balancing_constant .5 \
  --num_modes $NUM_MODES\
  --space_between_modes $SPACE_BETWEEN_MODES \
  --dist_var $VAR \
  --sampling_steps 10000 \
  --num_cycles 500 \
  --batch_size $BATCH_SIZE;

python multi_modal_ex.py \
  --sampler dmala \
  --step_size .014\
  --initial_balancing_constant .5 \
  --num_modes $NUM_MODES\
  --space_between_modes $SPACE_BETWEEN_MODES \
  --dist_var $VAR \
  --sampling_steps 10000 \
  --num_cycles 500 \
  --batch_size $BATCH_SIZE \
  --exponential_decay .999 \
  --min_lr .003 \
  --exponential_anneal;
