#!/bin/bash

python pcd.py --sampler $1 --use_balancing_constant --num_cycles 5 --include_exploration \
       	--burnin_adaptive --burnin_frequency $2 --cuda_id $3

