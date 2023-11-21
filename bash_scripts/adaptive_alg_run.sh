#!/bin/bash


python rbm_sample.py --samplers cyc_dula cyc_dmala --use_balancing_constant --burn_in_adaptive \
       	--save_dir ./figs/rbm_sample_adapt_hops_revbal  --num_seeds 10 --adaptive_bb_use_hops --adaptive_bb_reverse_bal \
	--adapt_alg big_is_better


python rbm_sample.py --samplers cyc_dula cyc_dmala --use_balancing_constant --burn_in_adaptive \
       	--save_dir ./figs/rbm_sample_adapt_nohops_revbal  --num_seeds 10  --adaptive_bb_reverse_bal \
	--adapt_alg big_is_better


python rbm_sample.py --samplers cyc_dula cyc_dmala --use_balancing_constant --burn_in_adaptive \
       	--save_dir ./figs/rbm_sample_adapt_hops_normbal  --num_seeds 10 --adaptive_bb_use_hops \
	--adapt_alg big_is_better

python rbm_sample.py --samplers cyc_dula cyc_dmala --use_balancing_constant --burn_in_adaptive \
       	--save_dir ./figs/rbm_sample_adapt_nohops_normbal  --num_seeds 10 \
	--adapt_alg big_is_better
