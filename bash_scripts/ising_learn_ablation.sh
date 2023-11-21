#!/bin/bash

for STEP_SIZE in 1.0 1.5 2.0
do
  for NUM_CYCLES in 5 10
  do
    for SAMPLER in 'cyc_dula' 'cyc_dmala'
    do
      for INIT_BAL in .9 1.0
        do
          python pcd.py --step_size $STEP_SIZE --sampler $SAMPLER --initial_balancing_constant $INIT_BAL \
          --use_balancing_constant --num_cycles $NUM_CYCLES --include_exploration
        done
      done
  done
done

python pcd.py --step_size .1 --sampler 'dmala'
python pcd.py --step_size .2 --sampler 'dmala'
python pcd.py --step_size .1 --sampler 'dula'
python pcd.py --step_size .2 --sampler 'dula'

