#!/bin/bash

for STEP_SIZE in .5 1.0 1.5 2.0
do
  for NUM_CYCLES in 50 100 250 500
  do
    for INIT_BAL in .6 .7 .8 .9 .99 1.0
    do
      python rbm_sample.py --step_size $STEP_SIZE --initial_balancing_constant $INIT_BAL \
      --use_balancing_constant --num_cycles $NUM_CYCLES --samplers cyc_dula cyc_dmala --include_exploration
    done
  done
done

python rbm_sample.py --step_size .1 --samplers dula dmala
python rbm_sample.py --step_size .2 --samplers dula dmala
