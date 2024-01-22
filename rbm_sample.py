import argparse
from posixpath import exists
import rbm
import torch
import numpy as np
import samplers
import mmd
import matplotlib.pyplot as plt
import os

import utils
import tensorflow_probability as tfp
import tqdm
import block_samplers
import time
from result_storing_utils import *
import pickle
from config_cmdline import (
    config_sampler_args,
    config_adaptive_args,
    config_SbC_args,
    potential_datasets,
)
from asbs_code.GBS.sampling.globally import AnyscaleBalancedSampler


def get_dmala_trained_rbm_sd(data, n_hidden):
    fn = f"figs/rbm_learn/{data}/itr_10000/{n_hidden}/dmala_stepsize_0.2_0.5/rbm_sd.pt"
    return torch.load(fn)


def get_gb_trained_rbm_sd(data, train_iter, rbm_name):
    fn = (
        f"figs/rbm_sample_res/{data}/zeroinit_False/rbm_iter_{train_iter}/{rbm_name}.pt"
    )
    print(fn)
    if os.path.isfile(fn):
        return torch.load(fn)
    return None


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
    makedirs(args.save_dir)
    device = torch.device(
        "cuda:" + str(args.cuda_id) if torch.cuda.is_available() else "cpu"
    )
    # seeds = utils.get_rand_seeds(args.seed_file)

    # if args.num_seeds == 1:
    #     seeds = [seeds[0]]
    # else:
    #     seeds = seeds[:min(len(seeds), args.num_seeds)]
    args.test_batch_size = args.batch_size
    args.train_batch_size = args.batch_size
    args.test_batch_size = args.batch_size
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.data in potential_datasets:
        assert args.n_visible == 784
        args.img_size = 28
        model = rbm.BernoulliRBM(args.n_visible, args.n_hidden)
        model.to(device)
        train_loader, test_loader, plot, viz = utils.get_data(args)

        init_data = []
        for x, _ in train_loader:
            init_data.append(x)
        init_data = torch.cat(init_data, 0)
        init_mean = init_data.mean(0).clamp(0.01, 0.99)
        model = rbm.BernoulliRBM(args.n_visible, args.n_hidden, data_mean=init_mean)
        if args.use_dmala_trained_rbm:
            sd = get_dmala_trained_rbm_sd(args.data, args.n_hidden)
            model.load_state_dict(sd)
            model.to(device)
            args.rbm_train_iter = "pretrain"
        else:
            rbm_name = f"rbm_lr_{str(args.rbm_lr)}_n_hidden_{str(args.n_hidden)}"
            sd = get_gb_trained_rbm_sd(args.data, args.rbm_train_iter, rbm_name)
            if sd is not None:
                print("Model found; omitting unnecessary training")
                model.load_state_dict(sd)
                model.to(device)
            else:
                print("No saved sd found, training rbm with gb")
                save_dir = f"figs/rbm_sample_res/{args.data}/zeroinit_False/rbm_iter_{args.rbm_train_iter}/"
                os.makedirs(save_dir, exist_ok=True)
                model.to(device)

                optimizer = torch.optim.Adam(model.parameters(), lr=args.rbm_lr)

                # xhat = model.init_dist.sample((args.n_test_samples,)).to(device)
                # train!
                itr = 0
                with tqdm.tqdm(total=args.rbm_train_iter) as pbar:
                    while itr < args.rbm_train_iter:
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

                            desc_str = "{} | log p(data) = {:.4f}, log p(model) = {:.4f}, diff = {:.4f}".format(
                                itr, d.mean(), m.mean(), (d - m).mean()
                            )

                            itr += 1

                            pbar.update(1)
                            if itr % args.print_every == 0:
                                pbar.set_description(desc_str, refresh=True)

                            # if itr % args.print_every == 0:
                            #     print(
                            #         "{} | log p(data) = {:.4f}, log p(model) = {:.4f}, diff = {:.4f}".format(
                            #             itr, d.mean(), m.mean(), (d - m).mean()
                            #         )
                            #     )
                    torch.save(model.state_dict(), save_dir + f"{rbm_name}.pt")
    else:
        model.W.data = torch.randn_like(model.W.data) * (0.05**0.5)
        model.b_v.data = torch.randn_like(model.b_v.data) * 1.0
        model.b_h.data = torch.randn_like(model.b_h.data) * 1.0
        viz = plot = None

    if args.get_base_energies:
        digit_energies = []
        digit_values = []
        for x, y in train_loader:
            x = x.to(device)
            d = model.logp_v_unnorm(x)
            digit_energies += list(d.detach().cpu().numpy())
            digit_values += list(y.cpu().numpy())
        digit_energy_res = {"energies": digit_energies, "values": digit_values}

    gt_samples = model.gibbs_sample(
        n_steps=args.gt_steps, n_samples=args.n_samples + args.n_test_samples, plot=True
    )
    kmmd = mmd.MMD(mmd.exp_avg_hamming, False)
    gt_samples, gt_samples2 = gt_samples[: args.n_samples], gt_samples[args.n_samples :]
    cur_dir_pre = f"{args.save_dir}/{args.data}/zeroinit_{args.zero_init}/rbm_iter_{args.rbm_train_iter}"
    os.makedirs(cur_dir_pre, exist_ok=True)
    if plot is not None:
        plot("{}/ground_truth.png".format(cur_dir_pre), gt_samples2)
    opt_stat = kmmd.compute_mmd(gt_samples2, gt_samples)
    print("gt <--> gt log-mmd", opt_stat, opt_stat.log10())
    pickle.dump(opt_stat.log10().cpu(), open(cur_dir_pre + "/gt_log_mmds.pt", "wb"))
    new_samples = model.gibbs_sample(n_steps=0, n_samples=args.n_test_samples)
    log_mmds = {}
    log_mmds["gibbs"] = []
    ars = {}
    hops = {}
    ess = {}
    times = {}
    chains = {}
    chain = []
    sample_var = {}

    x0 = model.init_dist.sample((args.n_test_samples,)).to(device)
    if args.zero_init:
        x0 = torch.zeros_like(x0).to(device)
    temps = args.samplers
    for temp in temps:
        if temp == "dim-gibbs":
            sampler = samplers.PerDimGibbsSampler(args.n_visible)
        elif temp == "rand-gibbs":
            sampler = samplers.PerDimGibbsSampler(args.n_visible, rand=True)
        elif "bg-" in temp:
            block_size = int(temp.split("-")[1])
            sampler = block_samplers.BlockGibbsSampler(args.n_visible, block_size)
        elif "hb-" in temp:
            block_size, hamming_dist = [int(v) for v in temp.split("-")[1:]]
            sampler = block_samplers.HammingBallSampler(
                args.n_visible, block_size, hamming_dist
            )
        elif temp == "gwg":
            sampler = samplers.DiffSampler(
                args.n_visible,
                1,
                fixed_proposal=False,
                approx=True,
                multi_hop=False,
                temp=2.0,
            )
        elif "gwg-" in temp:
            n_hops = int(temp.split("-")[1])
            sampler = samplers.MultiDiffSampler(
                args.n_visible, 1, approx=True, temp=2.0, n_samples=n_hops
            )
        elif "asb" in temp:
            sampler = AnyscaleBalancedSampler(
                args, cur_type="1st", sigma=0.1, alpha=0.5, adaptive=1
            )
            model_name = "anyscale"
        else:
            sampler = utils.get_dlp_samplers(temp, args.n_visible, device, args)

            model_name = sampler.get_name()
        cur_dir = f"{args.save_dir}/{args.data}/zeroinit_{args.zero_init}/rbm_iter_{args.rbm_train_iter}/{model_name}"
        if args.pair_optim:
            cur_dir += "_pair_opt"
        cur_dir += f"_{args.adapt_strat}"
        os.makedirs(cur_dir, exist_ok=True)
        x = x0.clone().detach()
        sample_var[temp] = []
        log_mmds[temp] = []
        ars[temp] = []
        hops[temp] = []
        times[temp] = []
        chain = []
        cur_time = 0.0
        print_every_i = 0
        if args.use_dula_init:
            print("using dula initialization")
            # only used for experiments with bDMALA and making
            # step size acc curve
            orig_sz = sampler.step_size
            dula_init_sz = 30  # magic number, point is just to make all inits the same
            sampler.step_size = dula_init_sz
            orig_mh = sampler.mh
            sampler.mh = False
            for i in range(100):
                x = sampler.step(x.detach(), model).detach()
            sampler.step_size = orig_sz
            sampler.mh = orig_mh

        if args.burnin_adaptive:
            if args.use_manual_EE:
                burnin_res = sampler.adapt_SbC(
                    x.detach(),
                    model,
                    budget=args.burnin_budget,
                    test_steps=args.burnin_test_steps,
                    init_big_step=30,
                    init_small_step=0.05,
                    init_big_bal=0.95,
                    init_small_bal=0.5,
                    lr=args.burnin_lr,
                    a_s_cut=args.a_s_cut,
                    use_dula=False,
                )
            else:
                if "cyc" in temp:
                    _, burnin_res = sampler.adapt_alg_greedy_mod(
                        x.detach(),
                        model,
                        budget=args.burnin_budget,
                        test_steps=args.burnin_test_steps,
                        init_big_step=30,  # TODO: make not hard coded
                        init_small_step=0.05,
                        init_big_bal=0.95,
                        init_small_bal=0.5,
                        lr=args.burnin_lr,
                        a_s_cut=args.a_s_cut,
                        bal_resolution=args.bal_resolution,
                        use_bal_cyc=False,
                    )
                else:
                    burnin_res = sampler.adapt_big_step(
                        x.detach(),
                        model,
                        budget=args.burnin_budget,
                        test_steps=args.burnin_test_steps,
                        lr=args.burnin_lr,
                        init_big_step=30,
                        init_bal=0.95,
                        a_s_cut=args.a_s_cut,
                        use_dula=not sampler.mh,
                    )
                    print(sampler.step_size)
                    print(sampler.bal)
            if "cyc" in temp:
                with open(f"{cur_dir}/burnin_res.pickle", "wb") as f:
                    burnin_res["final_steps"] = sampler.step_sizes.cpu().numpy()
                    burnin_res["final_bal"] = sampler.balancing_constants
                    pickle.dump(burnin_res, f)

        for i in tqdm.tqdm(range(args.n_steps), desc=f"{temp}"):
            # do sampling and time it
            st = time.time()
            if temp in ["cyc_dula", "cyc_dmala"]:
                xhat = sampler.step(x.detach(), model, i).detach()
            else:
                xhat = sampler.step(x.detach(), model).detach()
            cur_time += time.time() - st
            cur_hops = (x != xhat).float().sum(-1).mean().item()
            cur_sample_var = torch.var(xhat, dim=1)
            mean_var = torch.mean(cur_sample_var).detach().cpu().item()
            sample_var[temp].append(mean_var)
            # update trajectory
            x = xhat

            hops[temp].append(cur_hops)
            if i % args.subsample == 0:
                if args.ess_statistic == "dims":
                    chain.append(x.cpu().numpy()[0][None])
                else:
                    xc = x[0][None]
                    h = (xc != gt_samples).float().sum(-1)
                    chain.append(h.detach().cpu().numpy()[None])
            # if we are in the last cycle, just plot everything
            if (
                i >= args.n_steps - (int(args.n_steps / args.num_cycles))
                and plot is not None
            ):
                plot(f"{cur_dir}/sample_itr_{i}.png".format(args.save_dir, temp, i), x)
            else:
                if i % args.viz_every == 0 and plot is not None:
                    plot(
                        f"{cur_dir}/sample_itr_{i}.png".format(args.save_dir, temp, i),
                        x,
                    )

            if i % args.print_every == print_every_i:
                hard_samples = x
                stat = kmmd.compute_mmd(hard_samples, gt_samples)
                log_stat = stat.log().item()
                log_mmds[temp].append(log_stat)
                times[temp].append(cur_time)

        chain = np.concatenate(chain, 0)
        ess[temp] = get_ess(chain, args.burn_in)
        chains[temp] = chain
        ess_mean = ess[temp].mean()
        ess_std = ess[temp].std()

        print("ess = {} +/- {}".format(ess[temp].mean(), ess[temp].std()))
        # np.save("{}/rbm_sample_times_{}.npy".format(args.save_dir,temp),times[temp])
        # np.save("{}/rbm_sample_logmmd_{}.npy".format(args.save_dir,temp),log_mmds[temp])
        if temp in ["cyc_dula", "cyc_dmala", "dula", "dmala", "asb"]:
            ess_data = {"ess_mean": ess_mean, "ess_std": ess_std}
            with open(f"{cur_dir}/log_mmds.pickle", "wb") as f:
                pickle.dump(log_mmds[temp], f)

            with open(f"{cur_dir}/ess.pickle", "wb") as f:
                pickle.dump(ess_data, f)

            with open(f"{cur_dir}/sample_var.pickle", "wb") as f:
                pickle.dump(sample_var[temp], f)

            with open(f"{cur_dir}/hops.pickle", "wb") as f:
                pickle.dump(hops[temp], f)

            with open(f"{cur_dir}/times.pickle", "wb") as f:
                pickle.dump(times[temp], f)
            if args.get_base_energies:
                with open(f"{cur_dir}/digit_energies.pickle", "wb") as f:
                    pickle.dump(digit_energy_res, f)
            # store_sequential_data(cur_dir, model_name, "log_mmds", log_mmds[temp])
            # store_sequential_data(cur_dir, model_name, "times", times[temp])
            # write_ess_data(
            #     cur_dir, model_name, {"ess_mean": ess_mean, "ess_std": ess_std}
            # )

            if temp in ["dmala", "cyc_dmala"]:
                # store_sequential_data(args.save_dir, model_name, "a_s", sampler.a_s)
                with open(f"{cur_dir}/a_s.pickle", "wb") as f:
                    pickle.dump(sampler.a_s, f)
    plt.clf()
    for temp in temps:
        plt.plot(log_mmds[temp], label="{}".format(temp))

    plt.legend()
    plt.savefig("{}/logmmd.png".format(args.save_dir))


if __name__ == "__main__":
    potential_datasets = [
        "mnist",
        "fashion",
        "emnist",
        "caltech",
        "omniglot",
        "kmnist",
        "random",
    ]
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default="./figs/rbm_sample_res")
    parser.add_argument("--data", choices=potential_datasets, type=str, default="mnist")
    parser.add_argument("--n_steps", type=int, default=5000)
    parser.add_argument("--n_samples", type=int, default=500)
    parser.add_argument("--n_test_samples", type=int, default=100)
    parser.add_argument("--gt_steps", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=1234567)
    parser.add_argument("--use_dmala_trained_rbm", action="store_true")
    # rbm def
    parser.add_argument("--rbm_train_iter", type=int, default=1000)
    parser.add_argument("--n_hidden", type=int, default=500)
    parser.add_argument("--n_visible", type=int, default=784)
    parser.add_argument("--print_every", type=int, default=10)
    parser.add_argument("--viz_every", type=int, default=100)
    # for rbm training
    parser.add_argument("--rbm_lr", type=float, default=0.001)
    parser.add_argument("--cd", type=int, default=10)
    parser.add_argument("--img_size", type=int, default=28)
    parser.add_argument("--scheduler_buffer_size", type=int, default=100)
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=100)
    # for ess
    parser.add_argument("--subsample", type=int, default=1)
    parser.add_argument("--burn_in", type=float, default=0.1)
    parser.add_argument(
        "--ess_statistic", type=str, default="dims", choices=["hamming", "dims"]
    )
    parser.add_argument("--zero_init", action="store_true")
    parser.add_argument("--samplers", type=str, nargs="*", default=["cyc_dula"])
    # sampler params
    parser = config_sampler_args(parser)

    # adaptive hyper params
    parser = config_adaptive_args(parser)

    # sbc hyper params
    parser = config_SbC_args(parser)
    parser.add_argument("--get_base_energies", action="store_true")
    parser.add_argument("--cuda_id", type=int, default=0)
    parser.add_argument("--use_dula_init", action="store_true")
    args = parser.parse_args()

    main(args)
