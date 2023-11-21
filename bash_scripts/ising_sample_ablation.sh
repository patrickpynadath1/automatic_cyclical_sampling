#!/bin/bash

for STEP_SIZE in .5 1.0 1.5 2.0
do
  for NUM_CYCLES in 100 500 1000 5000
  do
    for INIT_BAL in .6 .7 .8 .9 .99 1.0
    do
      python ising_sample.py --step_size $STEP_SIZE --initial_balancing_constant $INIT_BAL \
      --use_balancing_constant --num_cycles $NUM_CYCLES --samplers cyc_dula --include_exploration
    done
  done
done

python ising_sample.py --step_size .1 --samplers dula dmala
python ising_sample.py --step_size .2 --samplers dula dmala