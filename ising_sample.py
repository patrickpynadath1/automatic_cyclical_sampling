import argparse
from result_storing_utils import *
import rbm
import torch
import numpy as np
import samplers
import matplotlib.pyplot as plt
import os
import torchvision
device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')
import tensorflow_probability as tfp
import block_samplers
import time
import neptune
import pickle
import tqdm
import itertools

def makedirs(dirname):
    """
    Make directory only if it's not already there.
    """
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def get_ess(chain, burn_in):
    c = chain
    l = c.shape[0]
    bi = int(burn_in * l)
    c = c[bi:]
    cv = tfp.mcmc.effective_sample_size(c).numpy()
    cv[np.isnan(cv)] = 1.
    return cv

def get_log_rmse(x,gt_mean):
    x = 2. * x - 1.
    x2 = ((x-gt_mean) ** 2).mean().sqrt()
    return x2.log().detach().cpu().numpy()

def tv(samples):
    gt_probs = np.load("{}/gt_prob_{}_{}.npy".format(args.save_dir,args.dim,args.bias))
    arrs, uniq_cnt = np.unique(samples, axis=0, return_counts=True)
    sample_probs = np.zeros_like(gt_probs)

    for i in range(arrs.shape[0]):
        sample_probs[i] = (uniq_cnt[i]*(1.)-1.)/samples.shape[0]
    l_dist =  np.abs((gt_probs - sample_probs)).sum()

def get_gt_mean(args,model):
    dim = args.dim**2
    A = model.J
    b = args.bias
    lst=torch.tensor(list(itertools.product([-1.0, 1.0], repeat=dim))).to(device)
    f = lambda x: torch.exp((x @ A * x).sum(-1)  + torch.sum(b*x,dim=-1))
    flst = f(lst)
    plst = flst/torch.sum(flst)
    gt_mean = torch.sum(lst*plst.unsqueeze(1).expand(-1,lst.size(1)),0)
    torch.save(gt_mean.cpu(),"{}/gt_mean_dim{}_sigma{}_bias{}.pt".format(args.save_dir,args.dim,args.sigma,args.bias))

    # gt_mean = torch.load("{}/gt_mean_dim{}_sigma{}_bias{}.pt".format(args.save_dir,args.dim,args.sigma,args.bias)).to(device)
    return gt_mean

def main(args):
    makedirs(args.save_dir)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model = rbm.LatticeIsingModel(args.dim, args.sigma, args.bias)
    model.to(device)
    gt_mean = get_gt_mean(args,model)

    plot = lambda p, x: torchvision.utils.save_image(x.view(x.size(0), 1, args.dim, args.dim),
                                                     p, normalize=False, nrow=int(x.size(0) ** .5))
    ess_samples = model.init_sample(args.n_samples).to(device)

    hops = {}
    ess = {}
    times = {}
    chains = {}
    means = {}
    rmses = {}
    x0 = model.init_dist.sample((args.n_test_samples,)).to(device)
    possible_temps = ['cyc_dula', 'cyc_dmala','hb-10-1', 'bg-1', 'gwg', 'dmala', 'dula']
    # temps = ['cyc_dula', 'cyc_dmala', 'dmala', 'dula']
    temps = args.samplers
    # only allow samplers in this list
    for s in temps:
        assert s in possible_temps

    for temp in temps:
        to_log = {}
        if temp == 'dim-gibbs':
            sampler = samplers.PerDimGibbsSampler(model.data_dim)
        elif temp == "rand-gibbs":
            sampler = samplers.PerDimGibbsSampler(model.data_dim, rand=True)
        elif temp == "lb":
            sampler = samplers.PerDimLB(model.data_dim)
        elif "bg-" in temp:
            block_size = int(temp.split('-')[1])
            sampler = block_samplers.BlockGibbsSampler(model.data_dim, block_size)
            to_log["block_size"] = block_size
        elif "hb-" in temp:
            block_size, hamming_dist = [int(v) for v in temp.split('-')[1:]]
            sampler = block_samplers.HammingBallSampler(model.data_dim, block_size, hamming_dist)
            to_log["block_size"] = block_size
            to_log["hamming_dist"] = hamming_dist

        elif temp == "gwg":
            sampler = samplers.DiffSampler(model.data_dim, 1,
                                           fixed_proposal=False, approx=True, multi_hop=False, temp=2.)


        elif "gwg-" in temp:
            n_hops = int(temp.split('-')[1])
            sampler = samplers.MultiDiffSampler(model.data_dim, 1,
                                                approx=True, temp=2., n_samples=n_hops)

        elif temp == "dmala":
            sampler = samplers.LangevinSampler(model.data_dim, 1,
                                           fixed_proposal=False, approx=True, multi_hop=False, temp=2., step_size=args.step_size, mh=True)

        elif temp == "dula":
            sampler = samplers.LangevinSampler(model.data_dim, 1,
                                           fixed_proposal=False, approx=True, multi_hop=False, temp=2., step_size=args.step_size, mh=False)


        elif temp == "cyc_dmala":
            sampler = samplers.CyclicalLangevinSampler(model.data_dim, n_steps=1, num_cycles=args.num_cycles,
                                                       initial_balancing_constant=float(args.initial_balancing_constant),
                                                       use_balancing_constant=args.use_balancing_constant,
                                                        fixed_proposal=False, approx=True, multi_hop=False, temp=2.,
                                                       mean_stepsize=args.step_size, mh=True,
                                                       num_iters=args.n_steps, device=device,
                                                       include_exploration=args.include_exploration,
                                                       half_mh=args.halfMH)

        elif temp == "cyc_dula":
            sampler = samplers.CyclicalLangevinSampler(model.data_dim, n_steps=1, num_cycles=args.num_cycles,
                                                       use_balancing_constant=args.use_balancing_constant,
                                                       initial_balancing_constant=args.initial_balancing_constant,
                                                       fixed_proposal=False, approx=True, multi_hop=False, temp=2.,
                                                       mean_stepsize=args.step_size, mh=False, num_iters=args.n_steps,
                                                       device=device, include_exploration=args.include_exploration)

        else:
            raise ValueError("Invalid sampler...")


        x = x0.clone().detach()
        times[temp] = []
        hops[temp] = []
        chain = []
        cur_time = 0.
        mean = torch.zeros_like(x)
        time_list = []
        rmses[temp] = []

        for i in tqdm.tqdm(range(args.n_steps), desc=f"{temp}"):
            # do sampling and time it
            st = time.time()
            if temp in ['cyc_dula', 'cyc_dmala']:
                xhat = sampler.step(x.detach(), model, i).detach()
            else:
                xhat = sampler.step(x.detach(), model).detach()
            cur_time += time.time() - st

            # compute hamming dist
            cur_hops = (x != xhat).float().sum(-1).mean().item()

            # update trajectory
            x = xhat

            mean = mean + x
            if i % args.subsample == 0:
                if args.ess_statistic == "dims":
                    chain.append(x.cpu().numpy()[0][None])
                else:
                    xc = x
                    h = (xc != ess_samples[0][None]).float().sum(-1)
                    chain.append(h.detach().cpu().numpy()[None])

            if i % args.viz_every == 0 and plot is not None:
                time_list.append(cur_time)
                rmse = get_log_rmse(mean / (i+1),gt_mean)
                rmses[temp].append(rmse)



            if i % args.print_every == 0:
                times[temp].append(cur_time)
                hops[temp].append(cur_hops)
        
        means[temp] = mean / args.n_steps
        chain = np.concatenate(chain, 0)
        chains[temp] = chain

        if temp in ['cyc_dula', 'cyc_dmala', 'dula', 'dmala']:
            model_name = sampler.get_name()
            store_sequential_data(args.save_dir, model_name, "log_rmses", rmses[temp])
            store_sequential_data(args.save_dir, model_name, "times", time_list)
            if not args.no_ess:
                run_ess = get_ess(chain, args.burn_in)
                ess[temp] = run_ess
                write_ess_data(args.save_dir, model_name, {'ess_mean': run_ess.mean(), 'ess_std':run_ess.std()})
            if temp in ['dmala', 'cyc_dmala']:
                store_sequential_data(args.save_dir, model_name, "a_s", sampler.a_s)
            print(model_name)


    plt.clf()
    for temp in temps:
        plt.plot(rmses[temp], label="{}".format(temp))
    plt.legend()
    plt.savefig("{}/log_rmse.png".format(args.save_dir))

    if not args.no_ess:
        ess_temps = temps
        plt.clf()
        ess_list = [ess[temp] for temp in ess_temps]
        plt.boxplot(ess_list, labels=ess_temps, showfliers=False)
        plt.savefig("{}/ess.png".format(args.save_dir))
        plt.clf()
        plt.boxplot([ess[temp] / times[temp][-1] / (1. - args.burn_in) for temp in ess_temps], labels=ess_temps, showfliers=False)
        plt.savefig("{}/ess_per_sec.png".format(args.save_dir))
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default="./figs/ising_sample")
    parser.add_argument('--n_steps', type=int, default=50000)
    parser.add_argument('--n_samples', type=int, default=2)
    parser.add_argument('--n_test_samples', type=int, default=2)
    parser.add_argument('--seed', type=int, default=1234567)
    # model def
    parser.add_argument('--dim', type=int, default=5)
    parser.add_argument('--sigma', type=float, default=.1)
    parser.add_argument('--bias', type=float, default=0.2)
    # logging
    parser.add_argument('--print_every', type=int, default=10)
    parser.add_argument('--viz_every', type=int, default=1000)
    # for rbm training
    parser.add_argument('--rbm_lr', type=float, default=.001)
    parser.add_argument('--cd', type=int, default=10)
    parser.add_argument('--img_size', type=int, default=28)
    parser.add_argument('--batch_size', type=int, default=100)
    # for ess
    parser.add_argument('--subsample', type=int, default=1)
    parser.add_argument('--burn_in', type=float, default=.1)
    parser.add_argument('--ess_statistic', type=str, default="dims", choices=["hamming", "dims"])
    parser.add_argument('--no_ess', action="store_true")
    parser.add_argument('--num_cycles', type=int, default=5000)
    parser.add_argument('--step_size', type=float, default=2.0)
    parser.add_argument('--samplers', nargs='*', type=str, default=['cyc_dmala'])
    parser.add_argument('--initial_balancing_constant', type=float, default=1.0)
    parser.add_argument('--use_balancing_constant', action='store_true')
    parser.add_argument('--include_exploration', action='store_true')
    parser.add_argument('--halfMH', action='store_true')

    args = parser.parse_args()
    main(args)