# Read Me 

This is an in progress repository exploring the use of a cyclical step schedule for discrete langevin proposals.

The following commands will run the experiments for various combinations of hyper-parameters

To set up directories, run setup.sh

To generate the data, run 

bash generate_data.sh 

To reproduce results for rbm sampling experiment, run the following: 

bash rbm_sample_ablation.sh

To reproduce results for ising sample, run the following: 

bash ising_sample_ablation.sh

To reproduce results for ising_learn_ablation, run the following: 

bash ising_learn_ablation.sh

To reproduce results for ebm experiments, run the following: 

bash ebm.sh
bash ebm_cyc.sh

Once these are finished running, you can generate all the relevant figures by running the following: 

python generate_cdlp_plots.py