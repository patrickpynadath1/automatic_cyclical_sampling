# Read Me 

This is the supplementary material for the submission titled: 

"Gradient-based Discrete Sampling with Automatic Cyclical Scheduling"

To run the experiments for rbm sampling, run the following bash scripts: 

bash bash_scripts/rbm_sample_all_dmala.sh {CUDA ID GOES HERE}
bash bash_scripts/rbm_mode_escape_all.sh {CUDA ID GOES HERE}

To run the experiments for EBM sampling, run:
bash bash_scripts/ebm_sample_all.sh {CUDA ID GOES HERE}

To get the data for the step size and acceptance rate on EBMs with a fixed beta, run: 
bash bash_scripts/ebm_stepsize_acceptance_curve.sh {CUDA ID GOES HERE}

To get the data for rbm learning, run the following: 
bash bash_scripts/rbm_learn_all.sh {CUDA ID GOES HERE}

To generate Figure 1-5, run 
python generate_figures_mainbody.py 

THIS WILL NOT WORK IF YOU HAVE NOT GENERATED THE DATA. 
To get the data for ebm learning, you must first run all the bash scripts in bash_scripts/ebm_scripts 
To evaluate the models, you must run all the scripts in bash scripts/ais_scripts. 
The ais results will be in the file labeled "log.txt" as well as printed on the screen.