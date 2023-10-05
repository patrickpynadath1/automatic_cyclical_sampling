import matplotlib.pyplot as plt
from result_storing_utils import *
import argparse
import math

base_dir_images = "figs/cyc_dlp_results"
base_dir_ising = "figs/ising_sample"
base_dir_rbm = "figs/rbm_sample"
base_dir_ising_learn = "figs/ising_learn"

STEPSIZES = [.5, 1.0, 1.5, 2.0]
BAL_CONSTANTS = [.6, .7, .8, .9, 1.0]
RBM_CYCLES = [50, 100, 500, 1000]
ISINGSAMPLE_CYCLES = [100, 500, 1000, 5000]



def generate_plots(exp):
    if exp == 'ising_sample':
        ising_sample_best_performance()
        log_rmse_cyc_dula_dct, log_rmse_cyc_dmala_dct = get_value_dcts('ising_sample', 'log_rmses')
        ising_sample_bal_comp(log_rmse_cyc_dula_dct, log_rmse_cyc_dmala_dct)
        ising_sample_cycle_comp(log_rmse_cyc_dula_dct, log_rmse_cyc_dmala_dct)
    elif exp == 'rbm_sample':
        rbm_sample_best_performance()
        log_mmds_cyc_dula_dct, log_mmds_cyc_dmala_dct = get_value_dcts('rbm_sample', 'log_mmds')
        rbm_sample_bal_comp(log_mmds_cyc_dula_dct, log_mmds_cyc_dmala_dct)
        rbm_sample_cycle_comp(log_mmds_cyc_dula_dct, log_mmds_cyc_dmala_dct)
        rbm_sample_bal_sensitivity()
        rbm_sample_acc_rate_cycle()
    elif exp == 'ising_learn':
        ising_learn_best_performance()
    elif exp == 'ebm_learn':
        print("Vizualization function not done yet")
    return


# figure 1
def ising_sample_best_performance():
    ising_sample_base = get_data(base_dir_ising,
                                 get_normal_model('dmala', .4),
                                 'log_rmses')
    cyc_dula_ising_sample = get_data(base_dir_ising,
                                     get_cyc_model('cyc_dula', 1000, 2.0, True, 1.0, True, False), 'log_rmses')
    cyc_dmala_ising_sample = get_data(base_dir_ising,
                                      get_cyc_model('cyc_dmala', 5000, 2.0, True, 1.0, True, False), 'log_rmses')

    ising_sample_base_times = get_data(base_dir_ising,
                                       get_normal_model('dmala', .4),
                                       'times')
    cyc_dula_ising_sample_times = get_data(base_dir_ising,
                                           get_cyc_model('cyc_dula', 1000, 2.0, True, 1.0, True, False), 'times')
    cyc_dmala_ising_sample_times = get_data(base_dir_ising,
                                            get_cyc_model('cyc_dmala', 5000, 2.0, True, 1.0, True, False), 'times')

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    x = [i * 1000 for i in range(50)]
    ax[0].plot(x, ising_sample_base, label='baseline dmala')
    ax[0].plot(x, cyc_dula_ising_sample, label='best cyc-DULA')
    ax[0].plot(x, cyc_dmala_ising_sample, label='best cyc-DMALA')
    ax[0].grid()
    ax[0].legend()
    plt.grid()
    ax[0].set_ylabel('log rmse')
    ax[0].set_xlabel('Iterations')

    ax[1].plot(ising_sample_base_times, ising_sample_base, label="baseline dmala")
    ax[1].plot(cyc_dula_ising_sample_times, cyc_dula_ising_sample, label="best cyc-DULA")
    ax[1].plot(cyc_dmala_ising_sample_times, cyc_dmala_ising_sample, label="best cyc-DMALA")
    ax[1].set_xlabel('Runtime (s)')
    fig.savefig(f"{base_dir_images}/ising_best_performance_comp.svg", format='svg')
    return

# figure 2
def ising_sample_best_ess():
    return


# figure 3
def rbm_sample_best_performance():
    ising_sample_base = get_data(base_dir_rbm,
                                 get_normal_model('dmala', .2),
                                 'log_mmds')

    cyc_dula_ising_sample = get_data(base_dir_rbm,
                                     get_cyc_model('cyc_dula', 500, 2.0, True, 1.0, True, False), 'log_mmds')

    cyc_dmala_ising_sample = get_data(base_dir_rbm,
                                      get_cyc_model('cyc_dmala', 500, 2.0, True, 1.0, True, False), 'log_mmds')

    ising_sample_base_times = get_data(base_dir_rbm,
                                       get_normal_model('dmala', .2),
                                       'times')

    cyc_dula_ising_sample_times = get_data(base_dir_rbm,
                                           get_cyc_model('cyc_dula', 500, 2.0, True, 1.0, True, False), 'times')

    cyc_dmala_ising_sample_times = get_data(base_dir_rbm,
                                            get_cyc_model('cyc_dmala', 500, 2.0, True, 1.0, True, False), 'times')

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    x = [i * 10 for i in range(500)]
    ax[0].plot(x, ising_sample_base, label='baseline dmala')
    ax[0].plot(x, cyc_dula_ising_sample, label='best cyc-DULA')
    ax[0].plot(x, cyc_dmala_ising_sample, label='best cyc-DMALA')
    ax[0].grid()
    ax[0].set_ylabel('log rmse')
    ax[0].set_xlabel('Iterations')

    ax[1].plot(ising_sample_base_times, ising_sample_base, label='baseline dmala')
    ax[1].plot(cyc_dula_ising_sample_times, cyc_dula_ising_sample, label='best cyc-DULA')
    ax[1].plot(cyc_dmala_ising_sample_times, cyc_dmala_ising_sample, label='best cyc-DMALA')
    ax[1].grid()
    ax[1].set_ylabel('log rmse')
    ax[1].set_xlabel('Runtime (s)')
    ax[0].legend()
    fig.savefig(f"{base_dir_images}/rbm_best_performance_comp.svg", format='svg')
    return


# figure 4
def ising_learn_best_performance():
    ising_learn_base = get_data(base_dir_ising_learn,
                                get_normal_model('dmala', .2),
                                'rmse')
    cyc_dula_ising_learn = get_data(base_dir_ising_learn,
                                    get_cyc_model('cyc_dula', 2, 2.0, True, 1.0, True), 'rmse')
    cyc_dmala_ising_learn = get_data(base_dir_ising_learn,
                                     get_cyc_model('cyc_dmala', 2, 2.0, True, 1.0, True), 'rmse')

    ising_learn_base_times = get_data(base_dir_ising_learn,
                                      get_normal_model('dmala', .2),
                                      'times')
    cyc_dula_ising_learn_times = get_data(base_dir_ising_learn,
                                          get_cyc_model('cyc_dula', 2, 2.0, True, 1.0, True), 'times')
    cyc_dmala_ising_learn_times = get_data(base_dir_ising_learn,
                                           get_cyc_model('cyc_dmala', 2, 2.0, True, 1.0, True), 'times')
    x = [i * 1000 for i in range(11)]

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(x, np.log(ising_learn_base), label='baseline dmala')
    ax[0].plot(x, np.log(cyc_dula_ising_learn), label='cyc-DULA')
    ax[0].plot(x, np.log(cyc_dmala_ising_learn), label='cyc-DMALA')

    ax[0].grid()
    ax[0].set_ylabel('log rmse')
    ax[0].set_xlabel('Iterations')

    ax[1].plot(ising_learn_base_times, np.log(ising_learn_base), label='baseline dmala')
    ax[1].plot(cyc_dula_ising_learn_times, np.log(cyc_dula_ising_learn), label='cyc-DULA')
    ax[1].plot(cyc_dmala_ising_learn_times, np.log(cyc_dmala_ising_learn), label='cyc-DMALA')

    ax[1].grid()
    ax[1].set_xlabel('Runtime (s)')
    ax[0].legend()
    fig.savefig(f"{base_dir_images}/ising_learn_best_performance.svg", format="svg")
    return


# figure 5
def ebm_learn_comparison():
    return


# figure 6
def ising_sample_bal_comp(log_rmse_cyc_dula_dct, log_rmse_cyc_dmala_dct):
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    cycles = 1000
    small_step = .5
    big_step = 2.0
    x = [i * 1000 for i in range(50)]
    for b in BAL_CONSTANTS:
        data = log_rmse_cyc_dula_dct[get_dct_key(small_step, b, cycles)]
        ax[0, 0].grid()
        ax[0, 0].plot(x, data, label=b)
        ax[0, 0].legend(loc='lower left')
        ax[0, 0].set_title("DULA with Step Size .5")
        ax[0, 0].set_ylabel("log RMSE")

        data = log_rmse_cyc_dula_dct[get_dct_key(big_step, b, cycles)]
        ax[0, 1].grid()
        ax[0, 1].plot(x, data, label=f"balance constant = {b}")
        ax[0, 1].set_title("DULA with Step Size 2.0")

        data = log_rmse_cyc_dmala_dct[get_dct_key(small_step, b, cycles)]
        ax[1, 0].grid()
        ax[1, 0].plot(x, data, label=f"balance constant = {b}")
        ax[1, 0].set_title("DMALA with Step Size .5")
        ax[1, 0].set_ylabel("log RMSE")
        ax[1, 0].set_xlabel("Iterations")

        data = log_rmse_cyc_dmala_dct[get_dct_key(big_step, b, cycles)]
        ax[1, 1].grid()
        ax[1, 1].plot(x, data, label=f"balance constant = {b}")
        ax[1, 1].set_title("DMALA with Step Size 2.0")
        ax[1, 1].set_xlabel("Iterations")
    plt.setp(ax, yticks=[0, -1, -2, -3, -4, -5])
    plt.subplots_adjust(hspace=0.4)
    fig.savefig(f"{base_dir_images}/ising_sample_bal_comparisons.svg", format='svg')
    return


# figure 7
def rbm_sample_bal_comp(log_mmds_cyc_dula_dct, log_mmds_cyc_dmala_dct):

    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    cycles = 500
    small_step = .5
    big_step = 2.0
    x = [i * 10 for i in range(500)]
    for b in BAL_CONSTANTS:
        data = log_mmds_cyc_dula_dct[get_dct_key(small_step, b, cycles)]
        ax[0, 0].grid()
        ax[0, 0].plot(x, data, label=b)
        ax[0, 0].legend(loc='lower left')
        ax[0, 0].set_title("DULA with Step Size .5")
        ax[0, 0].set_ylabel("log MMDS")

        data = log_mmds_cyc_dula_dct[get_dct_key(big_step, b, cycles)]
        ax[0, 1].grid()
        ax[0, 1].plot(x, data, label=f"balance constant = {b}")
        ax[0, 1].set_title("DULA with Step Size 2.0")

        data = log_mmds_cyc_dmala_dct[get_dct_key(small_step, b, cycles)]
        ax[1, 0].grid()
        ax[1, 0].plot(x, data, label=f"balance constant = {b}")
        ax[1, 0].set_title("DMALA with Step Size .5")
        ax[1, 0].set_ylabel("log MMDS")
        ax[1, 0].set_xlabel("Iterations")

        data = log_mmds_cyc_dmala_dct[get_dct_key(big_step, b, cycles)]
        ax[1, 1].grid()
        ax[1, 1].plot(x, data, label=f"balance constant = {b}")
        ax[1, 1].set_title("DMALA with Step Size 2.0")
        ax[1, 1].set_xlabel("Iterations")
    plt.setp(ax, yticks=[-7, -6, -5, -4, -3, -2])
    plt.subplots_adjust(hspace=0.4)
    fig.savefig(f"{base_dir_images}/rbm_sample_bal_comparisons.svg", format='svg')
    return


# figure 8
def rbm_sample_bal_sensitivity():
    cycles = 500
    step = 2.0
    balance_param = [.99, 1.0]
    x = [i * 10 for i in range(500)]
    for b in balance_param:
        cyc_dula = get_cyc_model('cyc_dula', cycles, step, True, b, True, False)
        data = get_data('figs/rbm_sample', cyc_dula, 'log_mmds')
        plt.plot(x, data, label=f'balance param {b}')
    plt.grid()
    plt.xlabel('Iterations')
    plt.ylabel("Log MMDS")
    plt.legend()
    plt.savefig(f"{base_dir_images}/rbm_bal_sensitivity.svg", format='svg')
    return


# figure 9
def rbm_sample_cycle_comp(log_mmds_cyc_dula_dct, log_mmds_cyc_dmala_dct):
    # sensitivities to cycles -- rbm sample
    few_cycles = 50
    many_cycles = 500
    steps = [.5, 1.0, 1.5, 2.0]
    bal_values = [.6, .7, .8, .9, 1.0]
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))

    for s in steps:
        for i, cycles in enumerate([few_cycles, many_cycles]):
            for j, temp in enumerate(['cyc-DULA', 'cyc-DMALA']):
                ax[j, i].grid()
                ax[j, i].set_title(f"{cycles} cycles {temp}")
                best_val = np.inf
                best_b = None
                if temp == 'cyc-DULA':
                    value_dct = log_mmds_cyc_dula_dct
                else:
                    value_dct = log_mmds_cyc_dmala_dct
                for b in bal_values:
                    data = value_dct[get_dct_key(s, b, cycles)]
                    mean = data[int(3 * len(data)) // 4:].mean()
                    if mean < best_val:
                        best_val = mean
                        best_b = b
                best_data = value_dct[get_dct_key(s, best_b, cycles)]
                ax[j, i].plot(best_data, label=f"step {s} bal {best_b}")
                ax[j, i].legend()
                ax[j, i].grid()
                if j == 1:
                    ax[j, i].set_xlabel("Iterations")
                if i == 0:
                    ax[j, i].set_ylabel("Log MMDs")

    fig.savefig(f"{base_dir_images}/rbm_cycle_comp.svg", format='svg')
    return


# not a figure in the write up, but just to confirm similar behavior
def ising_sample_cycle_comp(log_rmse_cyc_dula_dct, log_rmse_cyc_dmala_dct):
    few_cycles = 100
    many_cycles = 1000
    steps = [.5, 1.0, 1.5, 2.0]
    bal_values = [.7, .8, .9, 1.0]
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))

    for s in steps:
        for i, cycles in enumerate([few_cycles, many_cycles]):
            for j, temp in enumerate(['cyc-DULA', 'cyc-DMALA']):
                ax[j, i].set_title(f"{cycles} cycles {temp}")
                best_val = np.inf
                best_b = None
                if temp == 'cyc-DULA':
                    value_dct = log_rmse_cyc_dula_dct
                else:
                    value_dct = log_rmse_cyc_dmala_dct
                for b in bal_values:
                    data = value_dct[get_dct_key(s, b, cycles)]
                    mean = data[len(data) // 2:].mean()
                    if mean < best_val:
                        best_val = mean
                        best_b = b
                best_data = value_dct[get_dct_key(s, best_b, cycles)]
                ax[j, i].plot(best_data, label=f"step {s} bal {best_b}")
                ax[j, i].grid()
                ax[j, i].legend()
                if j == 1:
                    ax[j, i].set_xlabel("Iterations")
                if i == 0:
                    ax[j, i].set_ylabel("Log RMSE")
    fig.savefig(f"{base_dir_images}/ising_sample_cycle_comp.svg", format='svg')
    return


# figure 10
def rbm_sample_acc_rate_cycle():
    a_s_dct = {}
    log_mmds_dct = {}
    for s in STEPSIZES:
        for b in BAL_CONSTANTS:
            for c in RBM_CYCLES:
                k = f'step_{s}_bal_{b}_cycles_{c}'
                dmala_cyc = get_cyc_model('cyc_dmala', c,
                                          s,
                                          True,
                                          b,
                                          True,
                                          False)
                a_s = get_data('figs/rbm_sample', dmala_cyc, 'a_s')
                a_s_dct[k] = a_s
                mmds = get_data('figs/rbm_sample', dmala_cyc, 'log_mmds')
                log_mmds_dct[k] = mmds
    # step 1.0, bal .9, cyc 100 has same behavior as step 2.0, bal 1.0, cyc 100
    dmala = get_normal_model('dmala', .2)
    a_s_normal = get_data('figs/rbm_sample', dmala, 'a_s')
    cyc = 100

    total_iter = math.ceil(5000 / cyc)
    cyc_a_s_dct = {}
    # getting average a_s for cycle
    for b in [.7, 1.0]:
        for s in [.5, 2.0]:
            k = get_dct_key(s, b, cyc)
            a_s_cyc = a_s_dct[k]
            a_s_cyc_avg = np.zeros((cyc, total_iter))
            a_s_normal_avg = np.zeros((cyc, total_iter))
            for k_loop in range(cyc):
                a_s_cyc_avg[k_loop, :] = a_s_cyc[k_loop * total_iter: (k_loop + 1) * total_iter]
                a_s_normal_avg[k_loop, :] = a_s_normal[k_loop * total_iter: (k_loop + 1) * total_iter]
            cyc_a_s_dct[f'{s}_{b}'] = a_s_cyc_avg

    avg_range = (0, 100)
    fig, ax = plt.subplots(2, 2, figsize=(10, 8))
    x = [i for i in range(50)]
    for j, b in enumerate([.7, 1.0]):
        for i, s in enumerate([.5, 2.0]):
            a_s_cyc_avg = cyc_a_s_dct[f'{s}_{b}']
            a_s_cyc_toplot = a_s_cyc_avg[avg_range[0]:avg_range[1], :].mean(axis=0)
            a_s_normal_toplot = a_s_normal_avg[avg_range[0]:avg_range[1], :].mean(axis=0)

            a_s_cyc_upper = a_s_cyc_toplot + a_s_cyc_avg[avg_range[0]:avg_range[1], :].std(axis=0)
            a_s_cyc_lower = a_s_cyc_toplot - a_s_cyc_avg[avg_range[0]:avg_range[1], :].std(axis=0)

            a_s_normal_upper = a_s_normal_toplot + a_s_normal.std(axis=0)
            a_s_normal_lower = a_s_normal_toplot - a_s_normal.std(axis=0)
            ax[j, i].fill_between(x, a_s_cyc_lower, a_s_cyc_upper, label=f"Step Size {s}")
            ax[j, i].fill_between(x, a_s_normal_lower, a_s_normal_upper, label="dmala step .2")

            ax[j, i].legend()
            ax[j, i].set_title(f"Balance Constant {b}")
            ax[j, i].grid()

    ax[0, 0].set_yticks([.0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0])
    ax[0, 1].set_yticks([.0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0])
    ax[1, 0].set_yticks([.0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0])
    ax[1, 1].set_yticks([.0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0])
    ax[1, 0].set_ylabel('Acceptance Rate')
    ax[0, 0].set_ylabel('Acceptance Rate')
    fig.savefig(f"{base_dir_images}/acc_rate_rbm_cyc.svg", format='svg')
    return


def get_value_dcts(exp, metric):
    log_mmds_cyc_dula_dct = {}
    log_mmds_cyc_dmala_dct = {}
    for s in STEPSIZES:
        for b in BAL_CONSTANTS:
            cycles = RBM_CYCLES if exp == 'rbm_sample' else ISINGSAMPLE_CYCLES
            for c in cycles:
                k = get_dct_key(s, b, c)
                cyc_dula = get_cyc_model('cyc_dula', c, s, True, b, True, False)
                data = get_data(f'figs/{exp}', cyc_dula, metric)
                log_mmds_cyc_dula_dct[k] = data

                cyc_dmala = get_cyc_model('cyc_dmala', c, s, True, b, True, False)
                data = get_data(f'figs/{exp}', cyc_dmala, metric)
                log_mmds_cyc_dmala_dct[k] = data
    return log_mmds_cyc_dula_dct, log_mmds_cyc_dmala_dct


# helper function for dealing with metadata
def get_dct_key(step, bal, cyc):
    k = f'step_{step}_bal_{bal}_cycles_{cyc}'
    return k



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', nargs='*', type=str, default=['rbm_sample', 'ising_sample', 'ising_learn', 'ebm_learn'])
    args = parser.parse_args()
    exps_to_visualize = args.exp
    for exp in exps_to_visualize:
        generate_plots(exp)
