#!/bin/bash
python rbm_sample.py --samplers dmala dula --use_big --num_seeds 10

for STEP_SIZE in .1 .2 .5 1.0 1.5  
do
	for BAL_CONSTANT in .5 .6 .7 .8 .9  
	do 
		python rbm_sample.py --step_size $STEP_SIZE --samplers dmala dula \
			--initial_balancing_constant $BAL_CONSTANT --num_seeds 10
	done
done

for STEP_SIZE in 2.0 4.0 5.0 6.0 7.0 8.0 10.0 20.0 
do
	for BAL_CONSTANT in .9 .92 .95 .97 1.0   
	do 
		python rbm_sample.py --step_size $STEP_SIZE --samplers dmala dula \
			--initial_balancing_constant $BAL_CONSTANT --num_seeds 10
	done
done



