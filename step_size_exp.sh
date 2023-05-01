for STEP_SIZE in .2 .4 .5 1.0
do
  for SAMPLER in dula cyc_dula
  do
    python pcd.py --step_size $STEP_SIZE --sampler $SAMPLER --n_iters 1500 --viz_every 100
  done
done