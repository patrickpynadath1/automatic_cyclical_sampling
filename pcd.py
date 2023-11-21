import argparse
import rbm
import torch
import numpy as np
import samplers
import mmd
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import torchvision
import utils
from tqdm import tqdm
import pickle
import time
from result_storing_utils import *


def makedirs(dirname):
    """
    Make directory only if it's not already there.
    """
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def l1(module):
    loss = 0.0
    for p in module.parameters():
        loss += p.abs().sum()
    return loss


def main(args):
    device = torch.device(
        "cuda:" + str(args.cuda_id) if torch.cuda.is_available() else "cpu"
    )
    args.device = device
    makedirs(args.save_dir)
    logger = open("{}/log.txt".format(args.save_dir), "w")
    print(f"\n\n{args.sampler}\n\n")

    def my_print(s):
        print(s)
        logger.write(str(s) + "\n")

    torch.cuda.manual_seed_all(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # load existing data
    # uncomment this when done with data generating
    if args.data == "mnist" or args.data_file is not None:
        train_loader, test_loader, plot, viz = utils.get_data(args)
    # generate the dataset
    else:
        data, data_model = utils.generate_data(args)
        my_print("we have created your data, but what have you done for me lately?????")
        with open("{}/data.pkl".format(args.save_dir), "wb") as f:
            pickle.dump(data, f)
        if args.data_model == "er_ising":
            ground_truth_J = data_model.J.detach().cpu()
            with open("{}/J.pkl".format(args.save_dir), "wb") as f:
                pickle.dump(ground_truth_J, f)
        quit()

    if args.model == "lattice_potts":
        model = rbm.LatticePottsModel(
            int(args.dim), int(args.n_state), 0.0, 0.0, learn_sigma=True
        )
        buffer = model.init_sample(args.buffer_size)
    elif args.model == "lattice_ising":
        model = rbm.LatticeIsingModel(int(args.dim), 0.0, 0.0, learn_sigma=True)
        buffer = model.init_sample(args.buffer_size)
    elif args.model == "lattice_ising_3d":
        model = rbm.LatticeIsingModel(int(args.dim), 0.2, learn_G=True, lattice_dim=3)
        ground_truth_J = model.J.clone().to(device)
        model.G.data = torch.randn_like(model.G.data) * 0.01
        model.sigma.data = torch.ones_like(model.sigma.data)
        buffer = model.init_sample(args.buffer_size)
        plt.clf()
        plt.matshow(ground_truth_J.detach().cpu().numpy())
        plt.savefig("{}/ground_truth.png".format(args.save_dir))
    elif args.model == "lattice_ising_2d":
        model = rbm.LatticeIsingModel(
            int(args.dim), args.sigma, learn_G=True, lattice_dim=2
        )
        ground_truth_J = model.J.clone().to(device)
        model.G.data = torch.randn_like(model.G.data) * 0.01
        model.sigma.data = torch.ones_like(model.sigma.data)
        buffer = model.init_sample(args.buffer_size)
        plt.clf()
        plt.matshow(ground_truth_J.detach().cpu().numpy())
        plt.savefig("{}/ground_truth.png".format(args.save_dir))
    elif args.model == "er_ising":
        model = rbm.ERIsingModel(int(args.dim), 2, learn_G=True)
        model.G.data = torch.randn_like(model.G.data) * 0.01
        buffer = model.init_sample(args.buffer_size)
        with open(args.graph_file, "rb") as f:
            ground_truth_J = pickle.load(f)
            plt.clf()
            plt.matshow(ground_truth_J.detach().cpu().numpy())
            plt.savefig("{}/ground_truth.png".format(args.save_dir))
        ground_truth_J = ground_truth_J.to(device)
    elif args.model == "rbm":
        model = rbm.BernoulliRBM(args.dim, args.n_hidden)
        buffer = model.init_dist.sample((args.buffer_size,))
    elif args.model == "dense_potts":
        raise ValueError
    elif args.model == "dense_ising":
        raise ValueError
    elif args.model == "mlp":
        raise ValueError

    model.to(device)
    buffer = buffer.to(device)

    # make G symmetric
    def get_J():
        j = model.J
        return (j + j.t()) / 2

    if args.sampler == "gibbs":
        if "potts" in args.model:
            sampler = samplers.PerDimMetropolisSampler(
                model.data_dim, int(args.n_state), rand=False
            )
        else:
            sampler = samplers.PerDimGibbsSampler(model.data_dim, rand=False)
    elif args.sampler == "rand_gibbs":
        if "potts" in args.model:
            sampler = samplers.PerDimMetropolisSampler(
                model.data_dim, int(args.n_state), rand=True
            )
        else:
            sampler = samplers.PerDimGibbsSampler(model.data_dim, rand=True)
    elif args.sampler == "gwg":
        if "potts" in args.model:
            sampler = samplers.DiffSamplerMultiDim(
                model.data_dim, 1, approx=True, temp=2.0
            )
        else:
            sampler = samplers.DiffSampler(
                model.data_dim, 1, approx=True, fixed_proposal=False, temp=2.0
            )
    elif args.sampler in ["dmala", "dula", "cyc_dmala", "cyc_dula"]:
        sampler = utils.get_dlp_samplers(args.sampler, model.data_dim, device, args)
    else:
        assert "gwg-" in args.sampler
        n_hop = int(args.sampler.split("-")[1])
        if "potts" in args.model:
            raise ValueError
        else:
            sampler = samplers.MultiDiffSampler(
                model.data_dim, 1, approx=True, temp=2.0, n_samples=n_hop
            )

    my_print(device)
    my_print(model)
    my_print(buffer.size())
    my_print(sampler)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    itr = 0
    sigmas = []
    sq_errs = []
    total_burnin_res = []
    rmses = []
    start_time = time.time()
    time_list = []
    if args.use_manual_EE:
        sampler.step_sizes = [args.big_step] + (
            [args.small_step] * (args.steps_per_cycle - 1)
        )
        sampler.balancing_constants = [args.big_bal] + (
            [args.small_bal] * (args.steps_per_cycle - 1)
        )
        print(f"step sizes")
        print(sampler.step_sizes)
        print(f"bal constants")
        print(sampler.balancing_constants)
        sampler.iter_per_cycle = len(sampler.step_sizes)
    while itr < args.n_iters:
        for x in train_loader:
            x = x[0].to(device)
            sampling_steps = args.sampling_steps

            # run burnin
            if (itr % args.burnin_frequency == 0 or itr == 0) and args.burnin_adaptive:
                print("Adapting")

                # burn in for manual EE
                # I only want to either the first or the last step size for now
                # 3 options: adapt_big, adapt_small, adapt_both
                if args.manual_EE_adapt_mode == "big":
                    sampler.dim = args.dim**2
                    burnin_res = sampler.adapt_steps(
                        buffer.detach(),
                        model,
                        args.burnin_budget,
                        steps_obj="alpha_max",
                        lr=args.burnin_lr,
                        init_step_size=args.burnin_init_stepsize,
                        test_steps=args.burnin_test_steps,
                        a_s_cut=args.adapt_a_s_cut,
                        tune_stepsize=True,
                    )
                elif args.manual_EE_adapt_mode == "small":
                    burnin_res = sampler.adapt_steps(
                        buffer.detach(),
                        model,
                        args.burnin_budget,
                        steps_obj="alpha_max",
                        lr=0.9,
                        init_step_size=args.small_step,
                        tune_stepsize=True,
                    )
                else:
                    # haven't fully implemented the algorithm for
                    # adapting both at the same time
                    burnin_res = sampler.adapt_steps(
                        buffer.detach(),
                        model,
                        args.burnin_budget,
                        steps_obj="alpha_max",
                        lr=0.9,
                        init_step_size=args.big_step,
                        tune_stepsize=True,
                    )
                _, final_step_size, burnin_hist, _ = burnin_res

                burnin_hist["end_stepsize"] = sampler.step_sizes
                burnin_hist["end_bal"] = sampler.balancing_constants
                total_burnin_res.append(burnin_hist)
            if args.use_manual_EE:
                if itr % args.steps_per_cycle == 0:
                    # switching this to True so I can look at the a_s results for a_s
                    sampler.mh = True
                    sampling_steps = args.big_step_sampling_steps
                else:
                    if "dmala" in args.sampler:
                        sampler.mh = True
            for k in range(sampling_steps):
                if args.sampler in ["cyc_dula", "cyc_dmala"]:
                    if args.use_outer_loop:
                        buffer = sampler.step(
                            buffer.detach(), model, (itr % args.steps_per_cycle)
                        ).detach()
                    else:
                        buffer = sampler.step(buffer.detach(), model, k).detach()
                else:
                    buffer = sampler.step(buffer.detach(), model).detach()
            logp_fake = model(buffer).squeeze().mean()
            logp_real = model(x).squeeze().mean()

            obj = logp_real - logp_fake
            loss = -obj
            loss += args.l1 * get_J().abs().sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model.G.data *= (1.0 - torch.eye(model.G.data.size(0))).to(model.G)

            if itr % args.print_every == 0:
                my_print(
                    "({}) log p(real) = {:.4f}, log p(fake) = {:.4f}, diff = {:.4f}, hops = {:.4f}".format(
                        itr,
                        logp_real.item(),
                        logp_fake.item(),
                        obj.item(),
                        sampler._hops,
                    )
                )

                if args.model in ("lattice_potts", "lattice_ising"):
                    my_print(
                        "\tsigma true = {:.4f}, current sigma = {:.4f}".format(
                            args.sigma, model.sigma.data.item()
                        )
                    )
                else:
                    sq_err = ((ground_truth_J - get_J()) ** 2).sum()
                    rmse = ((ground_truth_J - get_J()) ** 2).mean().sqrt()
                    my_print(
                        "\t err^2 = {:.4f}, log rmse = {:.4f}".format(
                            sq_err, torch.log(rmse)
                        )
                    )

            if itr % args.viz_every == 0:
                running_time = time.time() - start_time
                time_list.append(running_time)
                if args.model in ("lattice_potts", "lattice_ising"):
                    sigmas.append(model.sigma.data.item())
                    plt.clf()
                    plt.plot(sigmas, label="model")
                    plt.plot([args.sigma for s in sigmas], label="gt")
                    plt.legend()
                    plt.savefig("{}/sigma.png".format(args.save_dir))
                else:
                    sq_err = ((ground_truth_J - get_J()) ** 2).sum()
                    sq_errs.append(sq_err.item())
                    plt.clf()
                    plt.plot(sq_errs, label="sq_err")
                    plt.legend()
                    plt.savefig("{}/sq_err.png".format(args.save_dir))

                    rmse = ((ground_truth_J - get_J()) ** 2).mean().sqrt()
                    rmses.append(rmse.item())

                    plt.clf()
                    plt.plot(rmses, label="rmse")
                    plt.legend()
                    plt.savefig("{}/rmse.png".format(args.save_dir))

                    plt.clf()
                    plt.matshow(get_J().detach().cpu().numpy())
                    plt.savefig("{}/model_{}.png".format(args.save_dir, itr))

                plot("{}/data_{}.png".format(args.save_dir, itr), x.detach().cpu())
                plot(
                    "{}/buffer_{}.png".format(args.save_dir, itr),
                    buffer[: args.batch_size].detach().cpu(),
                )

            itr += 1

            if itr > args.n_iters:
                if args.model in ("lattice_potts", "lattice_ising"):
                    final_sigma = model.sigma.data.item()
                    with open("{}/sigma.txt".format(args.save_dir), "w") as f:
                        f.write(str(final_sigma))
                else:
                    if args.sampler in ["cyc_dula", "cyc_dmala", "dula", "dmala"]:
                        if args.use_manual_EE:
                            model_name = f"{args.sampler}_manual_EE_ss_{args.sampling_steps}_big_{args.big_step}_small_{args.small_step}_{args.big_step_sampling_steps}"
                            if args.burnin_adaptive:
                                model_name += f"_adapt_{args.manual_EE_adapt_mode}"
                                model_name += f"_acc_cut_{args.adapt_a_s_cut}"
                        else:
                            model_name = sampler.get_name()
                        store_sequential_data(
                            args.save_dir, model_name, "sqerr", sq_errs
                        )
                        store_sequential_data(args.save_dir, model_name, "rmse", rmses)
                        store_sequential_data(
                            args.save_dir, model_name, "times", time_list
                        )
                        if args.sampler in ["dmala", "cyc_dmala"]:
                            store_sequential_data(
                                args.save_dir, model_name, "a_s", sampler.a_s
                            )
                        if args.burnin_adaptive:
                            with open(
                                f"{args.save_dir}/{model_name}_burnin_res", "wb"
                            ) as f:
                                pickle.dump(total_burnin_res, f)

                quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument("--save_dir", type=str, default="./figs/ising_learn")
    parser.add_argument("--data", type=str, default="random")
    parser.add_argument(
        "--data_file",
        type=str,
        default="./DATASETS/ising_dim_25_sigma_.25/data.pkl",
        help="location of pkl containing data",
    )
    parser.add_argument(
        "--graph_file", type=str, help="location of pkl containing graph"
    )  # ER
    # data generation
    parser.add_argument("--gt_steps", type=int, default=1000000)
    parser.add_argument("--n_samples", type=int, default=2000)
    parser.add_argument("--sigma", type=float, default=0.25)  # ising and potts
    parser.add_argument("--bias", type=float, default=0.0)  # ising and potts
    parser.add_argument("--n_out", type=int, default=3)  # potts
    parser.add_argument("--degree", type=int, default=2)  # ER
    parser.add_argument(
        "--data_model",
        choices=[
            "rbm",
            "lattice_ising",
            "lattice_potts",
            "lattice_ising_3d",
            "er_ising",
        ],
        type=str,
        default="lattice_ising",
    )
    # models
    parser.add_argument(
        "--model",
        choices=[
            "rbm",
            "lattice_ising",
            "lattice_potts",
            "lattice_ising_3d",
            "lattice_ising_2d",
            "er_ising",
        ],
        type=str,
        default="lattice_ising_2d",
    )
    # mcmc
    parser.add_argument("--sampler", type=str, default="dmala")
    parser.add_argument("--seed", type=int, default=123456)
    parser.add_argument("--approx", action="store_true")
    parser.add_argument("--sampling_steps", type=int, default=100)
    parser.add_argument("--buffer_size", type=int, default=256)

    #
    parser.add_argument("--n_iters", type=int, default=10000)

    parser.add_argument("--n_hidden", type=int, default=25)
    parser.add_argument("--dim", type=int, default=25)
    parser.add_argument("--n_state", type=int, default=3)
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--viz_batch_size", type=int, default=1000)
    parser.add_argument("--print_every", type=int, default=100)
    parser.add_argument("--viz_every", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--l1", type=float, default=0.01)
    parser.add_argument("--step_size", type=float, default=2.0)
    parser.add_argument("--initial_balancing_constant", type=float, default=0.5)
    parser.add_argument("--use_balancing_constant", action="store_true")
    parser.add_argument("--include_exploration", action="store_true")
    parser.add_argument("--halfMH", action="store_true")
    parser.add_argument("--save_rejects", action="store_true")
    parser.add_argument("--save_diff", action="store_true")
    parser.add_argument("--burn_in_adaptive", action="store_true")
    parser.add_argument("--adapt_rate", type=float, default=0.025)
    parser.add_argument("--param_adapt", type=str, default="bal")
    parser.add_argument("--use_big", type=bool, default=False)
    parser.add_argument("--num_seeds", type=int, default=1)
    parser.add_argument("--num_cycles", type=int, default=2)

    # burn in arguments for manualEE
    parser.add_argument("--burnin_adaptive", action="store_true")
    parser.add_argument("--burnin_frequency", type=int, default=1000)
    parser.add_argument("--burnin_test_steps", type=int, default=10)
    parser.add_argument("--burnin_budget", type=int, default=5000)
    parser.add_argument("--cuda_id", type=int, default=0)
    parser.add_argument("--use_burnin_start", action="store_true")
    # can either be big, small, both
    parser.add_argument("--manual_EE_adapt_mode", type=str, default="big")
    parser.add_argument("--burnin_lr", type=float, default=0.5)
    parser.add_argument("--use_outer_loop", action="store_true")
    parser.add_argument("--steps_per_cycle", type=int, default=10)
    parser.add_argument("--use_manual_EE", action="store_true")
    parser.add_argument("--big_step", type=float, default=1.0)
    parser.add_argument("--small_step", type=float, default=0.1)
    parser.add_argument("--big_step_sampling_steps", type=int, default=5)
    parser.add_argument("--adapt_a_s_cut", type=float, default=0.5)
    parser.add_argument("--big_bal", type=float, default=0.5)
    parser.add_argument("--small_bal", type=float, default=0.5)
    parser.add_argument("--burnin_init_stepsize", type=float, default=50)
    args = parser.parse_args()
    args.burnin_step_obj = (
        "alpha_max"  # default is always alpha max based on sampling results
    )
    if args.use_outer_loop:
        args.n_steps = args.steps_per_cycle
        args.outer_cycles = args.num_cycles
        args.num_cycles = 1
    else:
        args.n_steps = args.sampling_steps
    main(args)
