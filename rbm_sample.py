import argparse
import rbm
import torch
import numpy as np
import samplers
import mmd
import matplotlib.pyplot as plt
import os
device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')
import utils
import tensorflow_probability as tfp
import tqdm
import block_samplers
import time
from result_storing_utils import *
import pickle
import neptune

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


def main(args):
    makedirs(args.save_dir)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model = rbm.BernoulliRBM(args.n_visible, args.n_hidden)
    model.to(device)

    if args.data == "mnist":
        assert args.n_visible == 784
        train_loader, test_loader, plot, viz = utils.get_data(args)

        init_data = []
        for x, _ in train_loader:
            init_data.append(x)
        init_data = torch.cat(init_data, 0)
        init_mean = init_data.mean(0).clamp(.01, .99)

        model = rbm.BernoulliRBM(args.n_visible, args.n_hidden, data_mean=init_mean)
        model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.rbm_lr)

        # train!
        itr = 0
        for x, _ in train_loader:
            x = x.to(device)
            xhat = model.gibbs_sample(v=x, n_steps=args.cd)

            d = model.logp_v_unnorm(x)
            m = model.logp_v_unnorm(xhat)

            obj = d - m
            loss = -obj.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if itr % args.print_every == 0:
                print("{} | log p(data) = {:.4f}, log p(model) = {:.4f}, diff = {:.4f}".format(itr,d.mean(), m.mean(),
                                                                                               (d - m).mean()))

    else:
        model.W.data = torch.randn_like(model.W.data) * (.05 ** .5)
        model.b_v.data = torch.randn_like(model.b_v.data) * 1.0
        model.b_h.data = torch.randn_like(model.b_h.data) * 1.0
        viz = plot = None



    gt_samples = model.gibbs_sample(n_steps=args.gt_steps, n_samples=args.n_samples + args.n_test_samples, plot=True)
    kmmd = mmd.MMD(mmd.exp_avg_hamming, False)
    gt_samples, gt_samples2 = gt_samples[:args.n_samples], gt_samples[args.n_samples:]
    if plot is not None:
        plot("{}/ground_truth.png".format(args.save_dir), gt_samples2)
    opt_stat = kmmd.compute_mmd(gt_samples2, gt_samples)
    print("gt <--> gt log-mmd", opt_stat, opt_stat.log10())

    new_samples = model.gibbs_sample(n_steps=0, n_samples=args.n_test_samples)
    log_mmds = {}
    log_mmds['gibbs'] = []
    ars = {}
    hops = {}
    ess = {}
    times = {}
    chains = {}
    chain = []
    x0 = model.init_dist.sample((args.n_test_samples,)).to(device)
    neptune_pref = "rbm_sample"
    temps = args.samplers
    for temp in temps:
        if temp == 'dim-gibbs':

            sampler = samplers.PerDimGibbsSampler(args.n_visible)
        elif temp == "rand-gibbs":
            sampler = samplers.PerDimGibbsSampler(args.n_visible, rand=True)
        elif "bg-" in temp:
            block_size = int(temp.split('-')[1])
            sampler = block_samplers.BlockGibbsSampler(args.n_visible, block_size)
        elif "hb-" in temp:
            block_size, hamming_dist = [int(v) for v in temp.split('-')[1:]]
            sampler = block_samplers.HammingBallSampler(args.n_visible, block_size, hamming_dist)
        elif temp == "gwg":
            sampler = samplers.DiffSampler(args.n_visible, 1,
                                           fixed_proposal=False, approx=True, multi_hop=False, temp=2.)
        elif "gwg-" in temp:
            n_hops = int(temp.split('-')[1])
            sampler = samplers.MultiDiffSampler(args.n_visible, 1,
                                                approx=True, temp=2., n_samples=n_hops)

        elif temp == "dmala":
            sampler = samplers.LangevinSampler(args.n_visible, 1,
                                               fixed_proposal=False, approx=True, multi_hop=False, temp=2.,
                                               step_size=args.step_size, mh=True, store_reject=args.save_rejects)

        elif temp == "dula":
            sampler = samplers.LangevinSampler(args.n_visible, 1,
                                               fixed_proposal=False, approx=True, multi_hop=False, temp=2.,
                                               step_size=args.step_size, mh=False)

        elif temp == "cyc_dula":
            sampler = samplers.CyclicalLangevinSampler(args.n_visible, n_steps=1, num_cycles=args.num_cycles,
                                                        use_balancing_constant=args.use_balancing_constant,
                                                        initial_balancing_constant=args.initial_balancing_constant,
                                                        fixed_proposal=False, approx=True, multi_hop=False, temp=2.,
                                                        mean_stepsize=args.step_size, mh=False, num_iters=args.n_steps,
                                                       include_exploration=args.include_exploration,
                                                       device=device,
                                                       store_diff=args.save_diff,
                                                       burn_in_adaptive=args.burn_in_adaptive,
                                                       adapt_rate=args.adapt_rate,
                                                       adapt_alg=args.adapt_alg,
                                                       param_to_adapt=args.param_adapt)
        elif temp == "cyc_dmala":
            sampler = samplers.CyclicalLangevinSampler(args.n_visible, n_steps=1, num_cycles=args.num_cycles,
                                                       use_balancing_constant=args.use_balancing_constant,
                                                       initial_balancing_constant=args.initial_balancing_constant,
                                                       fixed_proposal=False, approx=True, multi_hop=False, temp=2.,
                                                       mean_stepsize=args.step_size, mh=True,
                                                       num_iters=args.n_steps,
                                                       include_exploration=args.include_exploration,
                                                       device=device,
                                                       half_mh=args.halfMH,
                                                       store_reject=args.save_rejects,
                                                       store_diff=args.save_diff,
                                                       burn_in_adaptive=args.burn_in_adaptive,
                                                       adapt_rate=args.adapt_rate,
                                                       adapt_alg=args.adapt_alg,
                                                       param_to_adapt=args.param_adapt)


        else:
            raise ValueError("Invalid sampler...")

        x = x0.clone().detach()

        log_mmds[temp] = []
        ars[temp] = []
        hops[temp] = []
        times[temp] = []
        chain = []
        cur_time = 0.
        print_every_i = 0
        rejects = []
        acc = []
        model_name = sampler.get_name()

        if temp in ['cyc_dula', 'cyc_dmala']:
            if args.burn_in_adaptive:
                if args.adapt_alg == 'simple_cycle':
                    steps, burnin_acc = sampler.run_burnin_cycle_adaptive(x0.detach(), model, args.adaptive_cycles,
                                                                   lr=args.adapt_rate, opt_acc = .3)

                elif args.adapt_alg == 'simple_iter':
                    steps, burnin_acc = sampler.run_burnin_iter_adaptive(x0.detach(), model, args.adaptive_cycles, lr=args.adapt_rate)
                elif args.adapt_alg == 'sun_ab':
                    steps, burnin_hops = sampler.run_burnin_sun(x0.detach(), model, args.adaptive_cycles)
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
            # if temp in ['cyc_dmala', 'dmala'] and args.save_rejects:
            #     if sampler.have_reject:
            #         rejects.append(sampler.reject_sample.cpu().numpy())
            #         acc.append(xhat.detach().cpu().numpy())
            # compute hamming dist
            cur_hops = (x != xhat).float().sum(-1).mean().item()

            # update trajectory
            x = xhat


            if i % args.subsample == 0:
                if args.ess_statistic == "dims":
                    chain.append(x.cpu().numpy()[0][None])
                else:
                    xc = x[0][None]
                    h = (xc != gt_samples).float().sum(-1)
                    chain.append(h.detach().cpu().numpy()[None])


            if i % args.viz_every == 0 and plot is not None:
                plot("{}/temp_{}_samples_{}.png".format(args.save_dir, temp, i), x)


            if i % args.print_every == print_every_i:
                hard_samples = x
                stat = kmmd.compute_mmd(hard_samples, gt_samples)
                log_stat = stat.log().item()
                log_mmds[temp].append(log_stat)
                times[temp].append(cur_time)
                hops[temp].append(cur_hops)

        chain = np.concatenate(chain, 0)
        ess[temp] = get_ess(chain, args.burn_in)
        chains[temp] = chain
        ess_mean = ess[temp].mean()
        ess_std = ess[temp].std()

        print("ess = {} +/- {}".format(ess[temp].mean(), ess[temp].std()))
        # np.save("{}/rbm_sample_times_{}.npy".format(args.save_dir,temp),times[temp])
        # np.save("{}/rbm_sample_logmmd_{}.npy".format(args.save_dir,temp),log_mmds[temp])
        if temp in ['cyc_dula', 'cyc_dmala', 'dula', 'dmala']:
            store_sequential_data(args.save_dir, model_name, "log_mmds", log_mmds[temp])
            store_sequential_data(args.save_dir, model_name, "times", times[temp])
            write_ess_data(args.save_dir, model_name, {'ess_mean': ess_mean, 'ess_std':ess_std})
            if args.burn_in_adaptive:
                store_sequential_data(args.save_dir, model_name, "steps_burnin", steps)
                if args.adapt_alg in ['simple_iter', 'simple_cycle']:
                    store_sequential_data(args.save_dir, model_name, "a_s_burnin", burnin_acc)
                if args.adapt_alg == 'sun_ab':
                    store_sequential_data(args.save_dir, model_name, "burnin_hops", burnin_hops)
            if args.save_diff:
                store_sequential_data(args.save_dir, model_name, "diffs", np.array(sampler.diff_values))
                store_sequential_data(args.save_dir, model_name, "flip_probs", np.array(sampler.flip_probs))

            # if temp in ['dmala', 'cyc_dmala']:
            #     store_sequential_data(args.save_dir, model_name, "a_s", sampler.a_s)
            #     if args.save_rejects:
            #         store_sequential_data(args.save_dir, model_name, "rejects", np.array(rejects))
            #         store_sequential_data(args.save_dir, model_name, "acc", np.array(acc))

    plt.clf()
    for temp in temps:
        plt.plot(log_mmds[temp], label="{}".format(temp))

    plt.legend()
    plt.savefig("{}/logmmd.png".format(args.save_dir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default="./figs/rbm_sample")
    parser.add_argument('--data', choices=['mnist', 'random'], type=str, default='mnist')
    parser.add_argument('--n_steps', type=int, default=5000)
    parser.add_argument('--n_samples', type=int, default=500)
    parser.add_argument('--n_test_samples', type=int, default=100)
    parser.add_argument('--gt_steps', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=1234567)
    # rbm def
    parser.add_argument('--n_hidden', type=int, default=500)
    parser.add_argument('--n_visible', type=int, default=784)
    parser.add_argument('--print_every', type=int, default=10)
    parser.add_argument('--viz_every', type=int, default=100)
    # for rbm training
    parser.add_argument('--rbm_lr', type=float, default=.001)
    parser.add_argument('--cd', type=int, default=10)
    parser.add_argument('--img_size', type=int, default=28)
    parser.add_argument('--batch_size', type=int, default=100)
    # for ess
    parser.add_argument('--subsample', type=int, default=1)
    parser.add_argument('--burn_in', type=float, default=.1)
    parser.add_argument('--ess_statistic', type=str, default="dims", choices=["hamming", "dims"])
    parser.add_argument('--num_cycles', type=int, default=250)
    parser.add_argument('--step_size', type=float, default=1.5)
    parser.add_argument('--samplers', nargs='*', type=str, default=['cyc_dula'])
    parser.add_argument('--initial_balancing_constant', type=float, default=1)
    parser.add_argument('--use_balancing_constant', action='store_true')
    parser.add_argument('--include_exploration', action='store_true')
    parser.add_argument('--halfMH', action='store_true')
    parser.add_argument('--save_rejects', action='store_true')
    parser.add_argument('--save_diff', action='store_true')
    parser.add_argument('--adaptive_cycles', type=int, default=150)
    parser.add_argument('--burn_in_adaptive', action='store_true')
    parser.add_argument('--adapt_rate', type=float, default=.025)
    parser.add_argument('--adapt_alg', type=str, default='sun_ab')
    parser.add_argument('--param_adapt', type=str, default='bal')
    args = parser.parse_args()

    main(args)
