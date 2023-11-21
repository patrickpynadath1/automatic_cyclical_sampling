import argparse
from result_storing_utils import *
import rbm
import utils
import torch
import numpy as np
import samplers
import matplotlib.pyplot as plt
import os
import torchvision
import tensorflow_probability as tfp
import block_samplers
import time
import tqdm
import itertools
device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')


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
    
    seeds = utils.get_rand_seeds(args.seed_file)
    if args.num_seeds == 1: 
        seeds = [seeds[0]]
    else:
        seeds = seeds[:min(len(seeds), args.num_seeds)]
    # instantiate dictionary for data bookkeeping here 
    bookkeeping = {}
    for cur_seed in seeds:
        seed_res = {}
        torch.manual_seed(cur_seed)
        np.random.seed(cur_seed)
        model = rbm.LatticeIsingModel(args.dim, args.sigma, args.bias)
        model.to(device)
        print("model sent to device")
        gt_mean = get_gt_mean(args,model)
        print("Got mean ")
       #plot = lambda p, x: torchvision.utils.save_image(x.view(x.size(0), 1, args.dim, args.dim),
       #                                                 p, normalize=False, nrow=int(x.size(0) ** .5))
        plot = None 
        ess_samples = model.init_sample(args.n_samples).to(device)

        times = {}
        chains = {}
        means = {}
        x0 = model.init_dist.sample((args.n_test_samples,)).to(device)
        possible_temps = ['cyc_dula', 'cyc_dmala','hb-10-1', 'bg-1', 'gwg', 'dmala', 'dula']
        # temps = ['cyc_dula', 'cyc_dmala', 'dmala', 'dula']
        temps = args.samplers
        # only allow samplers in this list
        for s in temps:
            assert s in possible_temps

        for temp in temps:
            temp_res = {}
            if temp == 'dim-gibbs':
                sampler = samplers.PerDimGibbsSampler(model.data_dim)
            elif temp == "rand-gibbs":
                sampler = samplers.PerDimGibbsSampler(model.data_dim, rand=True)
            elif temp == "lb":
                sampler = samplers.PerDimLB(model.data_dim)
            elif "bg-" in temp:
                block_size = int(temp.split('-')[1])
                sampler = block_samplers.BlockGibbsSampler(model.data_dim, block_size)
            elif "hb-" in temp:
                block_size, hamming_dist = [int(v) for v in temp.split('-')[1:]]
                sampler = block_samplers.HammingBallSampler(model.data_dim, block_size, hamming_dist)
            elif temp == "gwg":
                sampler = samplers.DiffSampler(model.data_dim, 1,
                                               fixed_proposal=False, approx=True, multi_hop=False, temp=2.)
            elif "gwg-" in temp:
                n_hops = int(temp.split('-')[1])
                sampler = samplers.MultiDiffSampler(model.data_dim, 1,
                                                    approx=True, temp=2., n_samples=n_hops)
            else: 
                sampler = utils.get_dlp_samplers(temp, model.data_dim ** 2, device, args)
            
            model_name = sampler.get_name()
            x = x0.clone().detach()
            times[temp] = []
            chain = []
            cur_time = 0.
            mean = torch.zeros_like(x)
            temp_res['log_rmse'] = []
            temp_res['times'] = []
            temp_res['hops'] = []
            temp_res['times_hops'] = []
            
            if temp in ['cyc_dula', 'cyc_dmala']:
                if args.burn_in_adaptive:
                    if args.adapt_alg == 'simple_cycle':
                        opt_acc = .5
                        steps, burnin_acc = sampler.run_burnin_cycle_adaptive(x0.detach(), model, args.adaptive_cycles,
                                                                       r=args.adapt_rate, opt_acc = opt_acc)
                        temp_res['burnin_acc'] = burnin_acc
                        temp_res['burnin_steps'] = steps  
                    elif args.adapt_alg == 'simple_iter':
                        steps, burnin_acc = sampler.run_burnin_iter_adaptive(x0.detach(), model, args.adaptive_cycles, lr=args.adapt_rate)

                        temp_res['burnin_acc'] = burnin_acc
                        temp_res['burnin_steps'] = steps  
                    elif args.adapt_alg == 'sun_ab':
                        steps, burnin_hops = sampler.run_burnin_sun(x0.detach(), model, args.adaptive_cycles)
                        temp_res['burnin_steps'] = steps 
                        temp_res['burnin_hops'] = burnin_hops
                    elif args.adapt_alg == 'big_is_better':
                        steps, bal = sampler.run_burn_in_bb(x0.detach(), model)
                        temp_res['burnin_steps'] = steps
                        temp_res['burnin_bal'] = bal 
                    else:
                        raise ValueError("Not implemented yet :/")

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
                    temp_res['times'].append(cur_time)
                    temp_res['log_rmse'].append(get_log_rmse(mean / (i+1),gt_mean))

                if i % args.print_every == 0:
                    temp_res['times_hops'].append(cur_time)
                    temp_res['hops'].append(cur_hops)
                     
            means[temp] = mean / args.n_steps
            chain = np.concatenate(chain, 0)
            chains[temp] = chain 
            run_ess = get_ess(chain, args.burn_in)
            temp_res["ess_res"] = {'ess_mean': run_ess.mean(), 'ess_std':run_ess.std()}
            if temp in ['dmala', 'cyc_dmala']:
                temp_res["a_s"] = sampler.a_s
            seed_res[model_name] = temp_res
            print(model_name)
        bookkeeping[cur_seed] = seed_res
    output = utils.seed_averaging(bookkeeping)
    store_seed_avg(output, args.num_seeds, "ising_sample")
    print("stored seed average")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default="./figs/ising_sample")
    parser.add_argument('--n_steps', type=int, default=50000)
    parser.add_argument('--n_samples', type=int, default=2)
    parser.add_argument('--n_test_samples', type=int, default=2)
    parser.add_argument('--seed_file', type=str, default='seed.txt')
    # model def
    parser.add_argument('--dim', type=int, default=7)
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
    parser.add_argument('--save_rejects', action='store_true')
    parser.add_argument('--save_diff', action='store_true')
    parser.add_argument('--adaptive_cycles', type=int, default=150)
    parser.add_argument('--burn_in_adaptive', action='store_true')
    parser.add_argument('--adapt_rate', type=float, default=.025)
    parser.add_argument('--adapt_alg', type=str, default='big_is_better')
    parser.add_argument('--param_adapt', type=str, default='bal')
    parser.add_argument('--use_big', action="store_true")
    parser.add_argument('--num_seeds', type=int, default=1)

    args = parser.parse_args()
    print(args.num_seeds)
    main(args)
