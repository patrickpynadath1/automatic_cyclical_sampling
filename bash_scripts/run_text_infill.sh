#!/bin/bash

python run_text_sampling.py --num_cycles 5 --samples_per_example 25 --n_steps 100 --inference_mask_p .5 --examples 10 --dataset_name grimm
python run_text_sampling.py --num_cycles 5 --samples_per_example 25 --n_steps 100 --inference_mask_p .5 --examples 10 --dataset_name sst2
