import argparse
import rbm
import torch
import numpy as np
import samplers
import mmd
import os
import utils
import tensorflow_probability as tfp
import tqdm
import block_samplers
import time

from result_storing_utils import *
import pickle


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
    cv[np.isnan(cv)] = 1.0
    return cv


def main(args):
    device = torch.device(
        "cuda:" + str(args.cuda_id) if torch.cuda.is_available() else "cpu"
    )
    makedirs(args.save_dir)
    seeds = utils.get_rand_seeds(args.seed_file)
    if args.num_seeds == 1:
        seeds = [seeds[0]]
    else:
        seeds = seeds[: min(len(seeds), args.num_seeds)]
    # instantiate dictionary for data bookkeeping here
    bookkeeping = {}

    sampler = utils.get_dlp_samplers(temp, args.n_visible, device, args)
    cur_dir = f"{args.save_dir}/{sampler.get_name()}"
    os.makedirs(cur_dir, exist_ok=True)
    for cur_seed in seeds:
        model_name = f"{args.temp}"
        model_name += f"_SbC_big_{args.big_step}_small_{args.small_step}"
        model_name += f"_ss_{args.sampling_steps}_total_{args.total_iterations}"
        makedirs(args.save_dir)
        torch.manual_seed(cur_seed)
        np.random.seed(cur_seed)

        model = rbm.BernoulliRBM(args.n_visible, args.n_hidden)
        model.to(device)
        seed_res = {}
        # getting the sampler to be used for training the rbm
        temp = args.temp

        sampler.step_sizes = [args.big_step] + (
            [args.small_step] * (args.steps_per_cycle - 1)
        )
        sampler.balancing_constants = [args.big_bal] + (
            [args.small_bal] * (args.steps_per_cycle - 1)
        )
        sampler.iter_per_cycle = args.steps_per_cycle
        total_burnin_metrics = []

        assert args.n_visible == 784
        train_loader, test_loader, plot, viz = utils.get_data(args)

        init_data = []
        for x, _ in train_loader:
            init_data.append(x)
        init_data = torch.cat(init_data, 0)
        init_mean = init_data.mean(0).clamp(0.01, 0.99)

        model = rbm.BernoulliRBM(args.n_visible, args.n_hidden, data_mean=init_mean)
        model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.rbm_lr)
        itr = 0
        total_hops = []
        xhat = model.init_dist.sample((args.n_test_samples,)).to(device)
        with tqdm.tqdm(total=args.total_iterations) as pbar:
            while itr < args.total_iterations:
                # train!
                for x, _ in train_loader:
                    x = x.to(device)

                    if itr % args.burnin_frequency == 0:
                        burnin_res = sampler.adapt_steps(
                            xhat.detach(),
                            model,
                            args.burnin_budget,
                            test_steps=args.burnin_test_steps,
                            steps_obj="alpha_max",
                            lr=args.burnin_lr,
                            a_s_cut=args.burnin_a_s_cut,
                            tune_stepsize=True,
                        )
                        _, final_step_size, burnin_hist, _ = burnin_res
                        total_burnin_metrics.append(burnin_hist)
                    for i in range(args.sampling_steps):
                        # for now, the goal is merely to measure how alpha max changes as the model learns
                        # the data distribution

                        if temp in ["cyc_dula", "cyc_dmala"]:
                            xhat_new = sampler.step(xhat.detach(), model, itr).detach()
                        else:
                            xhat_new = sampler.step(xhat.detach(), model).detach()

                        # compute the hops
                        cur_hops = (xhat_new != xhat).float().sum(-1).mean().item()
                        total_hops.append(cur_hops)
                        xhat = xhat_new

                    if itr % args.viz_every == 0:
                        plot(
                            f"{cur_dir}/samples_{itr}.png",
                            xhat,
                        )
                    d = model.logp_v_unnorm(x)
                    m = model.logp_v_unnorm(xhat)

                    obj = d - m
                    loss = -obj.mean()

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    itr += 1
                    pbar.update(1)
        seed_res["hops"] = total_hops
        seed_res["burnin_res"] = total_burnin_metrics
        seed_res["a_s"] = sampler.a_s
        bookkeeping[cur_seed] = seed_res
    pickle.dump(bookkeeping, open(f"{cur_dir}/total_results.pickle", "wb"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default="./figs/rbm_learn")
    parser.add_argument(
        "--data", choices=["mnist", "random"], type=str, default="mnist"
    )
    parser.add_argument("--n_steps", type=int, default=5000)
    parser.add_argument("--n_samples", type=int, default=499)
    parser.add_argument("--n_test_samples", type=int, default=100)
    parser.add_argument("--gt_steps", type=int, default=10000)
    parser.add_argument("--seed_file", type=str, default="seed.txt")
    # rbm def
    parser.add_argument("--n_hidden", type=int, default=500)
    parser.add_argument("--n_visible", type=int, default=784)
    parser.add_argument("--print_every", type=int, default=10)
    parser.add_argument("--viz_every", type=int, default=1000)
    # for rbm training
    parser.add_argument("--rbm_lr", type=float, default=0.001)
    parser.add_argument("--cd", type=int, default=10)
    parser.add_argument("--img_size", type=int, default=28)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--sampling_steps", type=int, default=10)
    parser.add_argument("--total_iterations", type=int, default=10000)
    # for ess
    parser.add_argument("--subsample", type=int, default=1)
    parser.add_argument("--burn_in", type=float, default=0.1)
    parser.add_argument(
        "--ess_statistic", type=str, default="dims", choices=["hamming", "dims"]
    )
    parser.add_argument("--num_cycles", type=int, default=250)
    parser.add_argument("--step_size", type=float, default=2.0)
    parser.add_argument("--samplers", nargs="*", type=str, default=["cyc_dmala"])
    parser.add_argument("--initial_balancing_constant", type=float, default=1.0)
    parser.add_argument("--use_big", action="store_true")
    parser.add_argument("--num_seeds", type=int, default=1)

    # burnin hyper-param arguments
    parser.add_argument("--temp", type=str, default="cyc_dmala")
    parser.add_argument("--burnin_frequency", type=int, default=1000)
    parser.add_argument("--burnin_budget", type=int, default=1000)
    parser.add_argument("--burnin_adaptive", action="store_true")
    parser.add_argument("--burnin_test_steps", type=int, default=10)
    parser.add_argument("--burnin_step_obj", type=str, default="alpha_max")
    parser.add_argument("--burnin_init_bal", type=float, default=0.95)
    parser.add_argument("--burnin_a_s_cut", type=float, default=0.5)
    parser.add_argument("--burnin_lr", type=float, default=0.5)
    parser.add_argument("--burnin_error_margin_a_s", type=float, default=0.01)
    parser.add_argument("--burnin_error_margin_hops", type=float, default=5)
    parser.add_argument("--burnin_alphamin_decay", type=float, default=0.9)
    parser.add_argument("--burnin_bal_resolution", type=int, default=6)
    parser.add_argument("--cuda_id", type=int, default=0)
    parser.add_argument("--steps_per_cycle", type=int, default=10)

    # sbc hyper params
    parser.add_argument("--big_step", type=float, default=0.2)
    parser.add_argument("--small_step", type=float, default=0.2)
    parser.add_argument("--small_bal", type=float, default=0.5)
    parser.add_argument("--big_bal", type=float, default=0.5)

    args = parser.parse_args()

    main(args)
