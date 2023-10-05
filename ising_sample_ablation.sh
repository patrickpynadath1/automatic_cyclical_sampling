for STEP_SIZE in 1.0 1.5 2.0
do
  for NUM_CYCLES in 250 500
  do
    for INIT_BAL in .99 1.0
    do
      python ising_sample.py --step_size $STEP_SIZE --initial_balancing_constant $INIT_BAL \
      --use_balancing_constant --num_cycles $NUM_CYCLES --samplers cyc_dula --include_exploration
    done
  done
done