import numpy as np
import pickle
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
