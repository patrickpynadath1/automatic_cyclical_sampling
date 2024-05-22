#!/bin/bash
for data in mnist kmnist emnist omniglot caltech
do
    python sample_learned_rbms.py --rbm_save_dir figs/rbm_learn/$data/itr_2000/500/gwg --data $data;
    python sample_learned_rbms.py --rbm_save_dir figs/rbm_learn/$data/itr_2000/500/dmala_stepsize_0.2_0.5 --data $data;
    python sample_learned_rbms.py --rbm_save_dir figs/rbm_learn/$data/itr_2000/500/GB_100 --data $data;
    python sample_learned_rbms.py --rbm_save_dir raw_exp_data/rbm_learn/$data/itr_2000/500/acs --data $data;
done

