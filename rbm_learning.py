import argparse
import rbm
import torch
import numpy as np
import samplers
import mmd
import ais
import os
from samplers.tuning_components import BayesOptimizer
import utils
import tensorflow_probability as tfp
import tqdm
import block_samplers
import time
from config_cmdline import (
    config_acs_args,
    config_acs_pcd_args,
    potential_datasets,
)
import pickle
import pandas as pd
import wandb
from torchvision.utils import make_grid


class BlockGibbsWrapper:
    def __init__(self):
        pass

    def step(self, x, rbm_model):
        return rbm_model.gibbs_sample(v=x, n_step=1)


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
    sqrt = lambda x: int(torch.sqrt(torch.Tensor([x])))

    def get_grid_img(x):
        return make_grid(
            x.view(x.size(0), 1, args.img_size, args.img_size),
            normalize=True,
            nrow=sqrt(x.size(0)),
        )

    device = torch.device(
        "cuda:" + str(args.cuda_id) if torch.cuda.is_available() else "cpu"
    )
    wandb.init(project="acs", group="rbm_learn", config=args)
    makedirs(args.save_dir)
    cur_seed = args.seed
    args.test_batch_size = args.batch_size
    args.train_batch_size = args.batch_size
    args.test_batch_size = args.batch_size
    # instantiate dictionary for data bookkeeping here
    bookkeeping = {}

    if args.sampler == "gb":
        model_name = f"GB_{args.cd}"
    elif args.sampler == "gwg":
        sampler = samplers.DiffSampler(
            args.n_visible,
            1,
            fixed_proposal=False,
            approx=True,
            multi_hop=False,
            temp=2.0,
        )
        model_name = "gwg"
    else:
        sampler = utils.get_dlp_samplers(args.sampler, args.n_visible, device, args)
        model_name = args.sampler
    cur_dir = f"{args.save_dir}/{args.data}/itr_{args.total_iterations}/{args.n_hidden}/{model_name}"
    os.makedirs(cur_dir, exist_ok=True)

    torch.manual_seed(cur_seed)
    np.random.seed(cur_seed)

    model = rbm.BernoulliRBM(args.n_visible, args.n_hidden)
    model.to(device)
    # getting the sampler to be used for training the rbm
    temp = args.sampler

    total_burnin_metrics = []

    assert args.n_visible == 784
    train_loader, test_loader, plot, viz = utils.get_data(args)

    init_data = []
    for x, _ in train_loader:
        init_data.append(x)
    init_data = torch.cat(init_data, 0)
    init_mean = init_data.mean(0).clamp(0.01, 0.99).to(device)

    model = rbm.BernoulliRBM(args.n_visible, args.n_hidden, data_mean=init_mean)
    model.to(device)
    wandb.watch(model, log_freq=100)
    print(init_mean.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.rbm_lr)
    itr = 0
    total_hops = []
    xhat = model.init_dist.sample((args.n_test_samples,)).to(device)
    if args.sampler == "acs":
        burnin_res = []

    def preprocess(data):
        return data

    total_ais_res = []
    burnin_metrics = {"alpha_max": [], "alpha_min": []}
    with tqdm.tqdm(total=args.n_steps) as pbar:
        init_alpha_max = 5
        init_alpha_min = 0.05
        running_max = []
        running_min = []
        while itr < args.total_iterations:
            # train!
            for x, _ in train_loader:
                x = x.to(device)
                if args.sampler == "gb":
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
                    wandb.log({"loss": loss.item()})
                    pbar.update(1)
                    if itr % args.print_every == 0:
                        pbar.set_description(desc_str, refresh=True)
                else:
                    sampling_steps = args.sampling_steps
                    if args.sampler == "acs":
                        orig_mh = sampler.mh
                        cycle_num = itr // sampler.iter_per_cycle
                        if itr % sampler.iter_per_cycle == 0:
                            sampling_steps = args.big_step_sampling_steps
                            sampler.mh = False
                        else:
                            sampling_steps = args.sampling_steps
                            sampler.mh = True

                        if (
                            itr % sampler.iter_per_cycle == 0
                            and cycle_num % args.adapt_every == 0
                        ):
                            big_step_budget = (
                                (args.sampling_steps - args.big_step_sampling_steps)
                                * args.adapt_every
                                * 2
                            )
                            # tune the big step
                            # for right now, the initial alpha max = 30
                            (
                                xhat_new,
                                new_alpha_max,
                                alpha_max_metrics,
                            ) = sampler.adapt_big_step(
                                xhat.detach(),
                                model,
                                budget=big_step_budget,
                                test_steps=1,
                                init_big_step=5,
                                a_s_cut=args.a_s_cut,
                                lr=args.burnin_lr,
                                init_big_bal=args.burnin_big_bal,
                                use_dula=True,
                            )
                            print(new_alpha_max)
                            burnin_metrics["alpha_max"].append(alpha_max_metrics)
                        elif itr % sampler.iter_per_cycle == 1:
                            # tune the small step
                            (
                                xhat_new,
                                new_alpha_min,
                                alpha_min_metrics,
                            ) = sampler.adapt_small_step(
                                xhat.detach(),
                                model,
                                budget=args.sampling_steps,
                                test_steps=1,
                                init_small_step=init_alpha_min,
                                a_s_cut=args.a_s_cut,
                                lr=args.burnin_lr,
                                init_small_bal=args.burnin_small_bal,
                                use_dula=False,
                            )
                            running_min.append(new_alpha_min)
                            init_alpha_min = np.mean(running_min)
                            init_alpha_min = max(
                                0.05, np.mean(running_min) - np.std(running_min)
                            )
                            burnin_metrics["alpha_min"].append(alpha_min_metrics)
                        else:
                            xhat_new = xhat.clone()
                            for i in range(sampling_steps):
                                xhat_new = sampler.step(
                                    xhat_new.detach(), model, itr
                                ).detach()
                    else:
                        xhat_new = xhat.clone()
                        for _ in range(args.sampling_steps):
                            xhat_new = sampler.step(xhat_new.detach(), model).detach()
                        # compute the hops
                    cur_hops = (xhat_new != xhat).float().sum(-1).mean().item()
                    total_hops.append(cur_hops)
                    xhat = xhat_new
                    if itr % args.viz_every == 0:
                        img = wandb.Image(get_grid_img(xhat))
                        # plot(
                        #     f"{cur_dir}/samples_{itr}.png",
                        #     xhat,
                        # )
                        img = wandb.log({"pcd_buffer": img})
                    d = model.logp_v_unnorm(x)
                    m = model.logp_v_unnorm(xhat)

                    obj = d - m
                    loss = -obj.mean()
                    wandb.log({"loss": loss})
                    if args.sampler in ["acs", "dmala"]:
                        mean_a_s = np.mean(sampler.a_s)
                        wandb.log({"mean_acceptance_rate": sampler.a_s})
                        sampler.a_s = []
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    itr += 1
                    wandb.log({"loss": loss.item(), "hops": cur_hops})
                    pbar.update(1)

        # bookkeeping["hops"] = total_hops
        # if args.burnin_adaptive:
        #     bookkeeping["burnin_res"] = burnin_metrics
        # if args.sampler not in ["gb", "gwg"]:
        #     bookkeeping["a_s"] = sampler.a_s

    digit_energies = []
    digit_values = []
    for x, y in test_loader:
        x = x.to(device)
        d = model.logp_v_unnorm(x)
        digit_energies += list(d.detach().cpu().numpy())
        digit_values += list(y.cpu().numpy())
    digit_energy_res = {"energies": digit_energies, "values": digit_values}

    df = pd.DataFrame(digit_energy_res)
    with open(f"{cur_dir}/digit_energies.pickle", "wb") as f:
        pickle.dump(df, f)
    with open(f"{cur_dir}/hops.pickle", "wb") as f:
        pickle.dump(total_hops, f)
    # evaluating via AIS
    def preprocess(data):
        return data

    gb_samples = model.gibbs_sample(n_steps=10000, n_samples=100, plot=True)
    plot(
        f"{cur_dir}/gb_samples.pdf",
        gb_samples,
    )
    model.to(device)
    logZ, train_ll, val_ll, test_ll, ais_samples = ais.evaluate(
        model,
        model.init_dist,
        None,
        train_loader,
        train_loader,
        test_loader,
        preprocess,
        device,
        args.eval_sampling_steps,
        args.test_batch_size,
        is_cyclical=False,
        is_rbm=True,
    )
    plot(
        f"{cur_dir}/ais_samples.pdf",
        xhat,
    )
    ais_res = {"logZ": logZ, "train_ll": train_ll, "val_ll": val_ll, "test_ll": test_ll}
    wandb.log(ais_res)
    print(cur_dir)
    pickle.dump(ais_res, open(f"{cur_dir}/ais_res.pickle", "wb"))
    pickle.dump(total_ais_res, open(f"{cur_dir}/running_ais_res.pickle", "wb"))
    pickle.dump(bookkeeping, open(f"{cur_dir}/total_results.pickle", "wb"))
    torch.save(model.state_dict(), f"{cur_dir}/rbm_sd.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_every", type=int, default=200)
    parser.add_argument("--save_dir", type=str, default="raw_exp_data/rbm_learn")
    parser.add_argument("--data", choices=potential_datasets, type=str, default="mnist")
    parser.add_argument("--n_steps", type=int, default=2000)
    parser.add_argument("--n_samples", type=int, default=499)
    parser.add_argument("--n_test_samples", type=int, default=100)
    parser.add_argument("--gt_steps", type=int, default=10000)
    parser.add_argument("--seed_file", type=str, default="seed.txt")
    parser.add_argument("--cuda_id", type=int, default=0)
    # rbm deff
    parser.add_argument("--n_hidden", type=int, default=500)
    parser.add_argument("--n_visible", type=int, default=784)
    parser.add_argument("--print_every", type=int, default=10)
    parser.add_argument("--viz_every", type=int, default=100)
    # for rbm training
    parser.add_argument("--rbm_lr", type=float, default=0.001)
    parser.add_argument("--cd", type=int, default=100)
    parser.add_argument("--img_size", type=int, default=28)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--sampling_steps", type=int, default=40)
    parser.add_argument("--total_iterations", type=int, default=2000)
    # for ess
    parser.add_argument("--seed", type=int, default=1234567)
    parser.add_argument("--subsample", type=int, default=1)
    parser.add_argument("--burn_in", type=float, default=0.1)
    parser.add_argument(
        "--ess_statistic", type=str, default="dims", choices=["hamming", "dims"]
    )
    parser.add_argument("--sampler", type=str, default="acs")
    parser.add_argument("--use_big", action="store_true")
    parser.add_argument("--num_seeds", type=int, default=1)
    parser.add_argument("--eval_sampling_steps", type=int, default=100000)
    parser.add_argument("--step_size", type=float, default=0.2)
    # burnin hyper-param arguments
    parser = config_acs_args(parser)
    # sbc hyper params
    parser = config_acs_pcd_args(parser)
    args = parser.parse_args()
    args.min_lr = None
    main(args)
