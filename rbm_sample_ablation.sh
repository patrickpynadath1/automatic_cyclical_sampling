for PARAM_ADAPT in 'bal' 'step'
do
  for ADAPT_ALG in 'simple_cycle' 'sun_ab' 'simple_iter'
  do
    for ADAPT_RATE in .001 .025 .05
    do
      for STEP_SIZE in 1.5 2.0
      do
        for NUM_CYCLES in 250
        do
          for INIT_BAL in 1.0
          do
            python rbm_sample.py --step_size $STEP_SIZE --initial_balancing_constant $INIT_BAL \
            --use_balancing_constant --num_cycles $NUM_CYCLES --samplers cyc_dula --include_exploration --burn_in_adaptive \
            --adapt_rate $ADAPT_RATE --adapt_alg $ADAPT_ALG --param_adapt $PARAM_ADAPT
          done
        done
      done
    done
  done
done
