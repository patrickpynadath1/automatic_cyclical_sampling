import numpy as np
import pickle
import os
# commands for making it easier to store data

def get_cyc_model(temp,
                  num_cycles,
                  step_size,
                  use_balancing_constant,
                  initial_balancing_constant,
                  include_exploration,
                  half_mh,
                  burn_in=False,
                  adapt_rate=.01,
                  adapt_alg='simple_cyc',
                  adapt_param = 'step'):
    if half_mh:
        name= f"{temp}_cycles_{num_cycles}_stepsize_{step_size}" + \
               f"_usebal_{use_balancing_constant}_initbal_{initial_balancing_constant}" + \
               f"_include_exploration_{include_exploration}_halfMH"
    else:
        name= f"{temp}_cycles_{num_cycles}_stepsize_{step_size}" + \
               f"_usebal_{use_balancing_constant}_initbal_{initial_balancing_constant}" + \
               f"_include_exploration_{include_exploration}"
    if burn_in:
        name += f"_burnin_adaptive_{adapt_rate}_alg_{adapt_alg}_param_{adapt_param}"
    return name


def get_normal_model(temp, stepsize):
    return f"{temp}_stepsize_{stepsize}"



def store_seed_avg(seed_avg, num_seeds, exp):
    base_dir = f"{exp}/num_seeds_{num_seeds}"
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)
    for model_name, res in seed_avg.items():
        for metric, statistics in res.items():
            if metric != "ess_res" and "burnin" not in metric:
                store_sequential_data(base_dir, model_name, f"{metric}_mean", statistics['mean'])

                store_sequential_data(base_dir, model_name, f"{metric}_var", statistics['var'])
            elif "burnin" in metric: 
                file_name = f"{base_dir}/{model_name}_{metric}.pickle"
                with open(file_name, 'wb') as file:
                    pickle.dump(statistics, file)
            else:
                write_ess_data(base_dir, model_name, statistics)
    return 


def package_temp_results(temp, hops, log_mmds, run_ess, times, sampler,
                         steps = None, burnin_acc = None, burnin_hops = None):
    temp_res = {}
    temp_res["hops"] = hops[temp]
    temp_res["log_mmds"] = log_mmds[temp]
    temp_res["times"] = times[temp]  
    temp_res["ess_res"] = {'ess_mean': run_ess.mean(), 'ess_std':run_ess.std()}
    if temp in ['dmala', 'cyc_dmala']:
        temp_res["a_s"] = sampler.a_s
        temp_res["steps_burnin"] = steps
    if sampler.burn_in_adaptive and burnin_acc and burnin_hops: # just need to make sure they are not None  
        if sampler.adapt_alg in ['simple_iter', 'simple_cycle']:
            temp_res["burnin_acc"] = burnin_acc
        if sampler.adapt_alg == 'sun_ab':
            temp_res["burnin_hops"] = burnin_hops
        temp_res["burn_in_flip_probs"] = sampler.burn_in_flip_probs
        temp_res["flip_probs"] = sampler.flip_probs
    return temp_res


# function for getting data goes here

# function for saving data goes here
def get_file_name(base_dir,
                  model_name,
                  metric_name):
    return f"{base_dir}/{model_name}_{metric_name}.npy"



def store_sequential_data(base_dir,
                          model_name,
                          metric_name,
                          metric):
    file_name = get_file_name(base_dir, model_name, metric_name)
    np.save(file_name, metric)
    return


def get_data(base_dir, model_name, metric_name):
    file_name = get_file_name(base_dir, model_name, metric_name)
    return np.load(file_name, allow_pickle=True)


# function for saving ess/scalar value functions go here
def write_ess_data(base_dir, model_name, value_dct):
    file_name = f"{base_dir}/{model_name}_ess_res.pickle"
    with open(file_name, 'wb') as file:
        pickle.dump(value_dct, file)
    return


def retrieve_ess_data(base_dir, model_name):
    file_name = f"{base_dir}/{model_name}_ess_res.pickle"
    with open(file_name, 'rb') as file:
        return pickle.load(file)
