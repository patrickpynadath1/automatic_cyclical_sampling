import argparse
import torch
import numpy as np
import vamp_utils
import os
import tensorflow_probability as tfp
import tqdm
import mlp
import random
from samplers import LangevinSampler
from pcd_ebm_ema import get_sampler, EBM
import torchvision
from rbm_sample import get_ess
import mmd
from config_cmdline import config_acs_args, config_acs_pcd_args
import pickle
import time
import pandas as pd
import wandb
from torchvision.utils import make_grid


def sqrt(x):
    int(torch.sqrt(torch.Tensor([x])))


def main(args):
    sqrt = lambda x: int(torch.sqrt(torch.Tensor([x])))
    def get_grid_img(x): 
        return make_grid(
            x.view(x.size(0), 1, args.img_size, args.img_size),
            normalize=True,
            nrow=sqrt(x.size(0)),
        )
    wandb.init(project="acs", group="ebm_sample", config=args)
    device = torch.device(
        "cuda:" + str(args.cuda_id) if torch.cuda.is_available() else "cpu"
    )
    args.device = device

    # plot = lambda p, x: torchvision.utils.save_image(
    #     x.view(x.size(0), args.input_size[0], args.input_size[1], args.input_size[2]),
    #     p,
    #     normalize=True,
    #     nrow=int(x.size(0) ** 0.5),
    # )

    def my_print(s):
        print(s)

    my_print("Loading data")
    train_loader, test_loader, val_loader, args = vamp_utils.load_dataset(args)

    def preprocess(data):
        if args.dynamic_binarization:
            return torch.bernoulli(data)
        else:
            return data


    my_print("Making Model")
    if args.model.startswith("mlp-"):
        nint = int(args.model.split("-")[1])
        net = mlp.mlp_ebm(np.prod(args.input_size), nint)
    elif args.model.startswith("resnet-"):
        nint = int(args.model.split("-")[1])
        net = mlp.ResNetEBM(nint)
    elif args.model.startswith("cnn-"):
        nint = int(args.model.split("-")[1])
        net = mlp.MNISTConvNet(nint)
    else:
        raise ValueError("invalid model definition")

    sampler = get_sampler(args)

    sampler_name = args.sampler
    if sampler_name == "asb":
        # the A paper did not use adaptive for this task
        sampler.adaptive = 0
        sampler_name += f"_use_diag_{args.use_diag_acs}"
    if args.use_acs_ebm:
        cur_dir = f"{args.save_dir}/{args.dataset_name}_acs_ebm/{sampler_name}"
    else:
        cur_dir = f"{args.save_dir}/{args.dataset_name}/{sampler_name}"
    os.makedirs(cur_dir, exist_ok=True)

    # copying the same initialization for the buffer as in pcd_ebm_ema
    init_batch = []
    for x, _ in val_loader:
        init_batch.append(x.to(device))
    init_batch = torch.cat(init_batch, 0)
    eps = 1e-2
    init_mean = init_batch.mean(0) * (1.0 - 2 * eps) + eps
    init_var = init_batch.std(0) ** 2 * (1 - 4 * eps) + eps
    init_dist = torch.distributions.Bernoulli(probs=init_mean)
    pure_rand = torch.distributions.Bernoulli(probs=torch.ones_like(init_mean).to(device) * .5)
    args.base_dist = True
    if args.base_dist:
        model = EBM(net, init_mean, temp=args.ebm_temp)
    else:
        model = EBM(net)
    model = model.to(device)
    energies = []
    for x, _ in train_loader:
        x = preprocess(x)
        energies += list(model(x.to(device)).cpu().detach().numpy())
    print(np.mean(energies))
    print(np.std(energies))


    average_differences = []
    mean_diffs = []

    # dataset analysis
    for x, _ in test_loader:
        x = preprocess(x)
        for i in range(10):
            x = x.to(device) 
            indices_to_sample = random.sample(range(len(x)), 2)
            i, j = indices_to_sample[0], indices_to_sample[1]
            diff = (x[i, :] != x[j, :]).float().sum(-1).mean().item()
            average_differences.append(diff)
            m1 = init_dist.sample((x.size(0), )).to(device)
            m2 = init_dist.sample((args.samples_to_generate,)).to(device)

            diff = (m1 != x).float().sum(-1).mean().item()
            mean_diffs.append(diff)
    print(np.mean(average_differences))
    print(np.mean(mean_diffs))
    if args.use_acs_ebm:
        ckpt_path = "figs/ebm/cs/caltech_0.7_30/best_ckpt_caltech_cyc_dmala_1.5.pt"
    else:
        ckpt_path = (
            f"example_ebms/gwg_ebm_{args.dataset_name}.pt"
        )
    print(ckpt_path)
    d = torch.load(ckpt_path)
    model.load_state_dict(d["ema_model"])
    model.L = 1
    x_init = init_dist.sample((args.samples_to_generate,)).to(device)
    print(x_init.shape)
    print(model(x_init))
    gt = init_dist.sample((args.samples_to_generate * 2,)).to(device)
    if args.zero_init:
        x_init = torch.zeros_like(x_init).to(device)
    model = model.to(device)

    #     print(starting_batch.shape)
    # os.makedirs(cur_dir, exist_ok=True)
    # if plot is not None:
    #     plot("{}/ground_truth.png".format(cur_dir_pre), gt_samples2)
    # opt_stat = kmmd.compute_mmd(gt_samples2, gt_samples)
    # print("gt <--> gt log-mmd", opt_stat, opt_stat.log10())
    # pickle.dump(opt_stat.log10().cpu(), open(cur_dir_pre + "/gt_log_mmds.pt", "wb"))

    # metrics to keep track of:
    energies = []
    hops = []
    sample_var = []
    acceptance_rates = []
    log_mmds = []
    times = []
    hops = []
    # if args.sampler == "acs":
    #     if args.use_diag_acs: 
    #         sampler.D = init_var.to(device).repeat(args.samples_to_generate, 1)**(-.1) 
    #         print(sampler.D)
    #         print(sampler.D.max())
    #         print(sampler.D.min())
    #     x_init, burnin_res = sampler.tuning_alg(
    #         x_init.detach(),
    #         model,
    #         budget=args.burnin_budget // 2,
    #         test_steps=1,
    #         init_big_step=5,
    #         init_small_step=0.05,
    #         init_big_bal=args.burnin_big_bal,
    #         init_small_bal=args.burnin_small_bal,
    #         lr=args.burnin_lr,
    #         a_s_cut=args.a_s_cut,
    #         bal_resolution=args.bal_resolution,
    #         use_bal_cyc=False,
    #     )
    if args.norm_mterm: 
        sampler.norm_mterm = 1
    else: 
        sampler.norm_mterm = None

    if args.sampler == 'acs': 
        x_init, burnin_res = sampler.tuning_alg(
            x_init.detach(),
            model,
            budget=args.burnin_budget,
            test_steps=1,
            init_big_step=5,
            init_small_step=0.05,
            init_big_bal=args.burnin_big_bal,
            init_small_bal=args.burnin_small_bal,
            lr=args.burnin_lr,
            a_s_cut=args.a_s_cut,
            bal_resolution=args.bal_resolution,
            use_bal_cyc=False,
            dula_burnin=args.dula_burnin_steps,
            acs_burnin=args.acs_burnin_steps
        )
        with open(f"{cur_dir}/burnin_res.pickle", "wb") as f:
            pickle.dump(burnin_res, f)
            print(cur_dir)

    if args.sampler == "acs":
        print(f"steps: {sampler.step_sizes}")
        print(f"bal: {sampler.balancing_constants}")
    itr_per_cycle = args.sampling_steps // args.num_cycles
    if args.sampler == "asb":
        if args.use_diag_acs: 
            sampler.D = init_var.to(device) ** (-0.2) / 0.1
        else:
            sampler.D = torch.ones_like(init_var).to(device) / 0.1

    cur_time = 0.0
    total_iter = args.sampling_steps 
    for itr in tqdm.tqdm(range(total_iter)):
        st = time.time()
        if args.sampler == "acs":
            x_cur = sampler.step(x_init, model, itr)
        else:
            x_cur = sampler.step(x_init, model)

        cur_time += time.time() - st
        cur_hops = (x_cur != x_init).float().sum(-1).mean().item()
        # calculate the hops
        if itr % args.print_every == 0:
            hard_samples = x_cur
            times.append(cur_time)

        hops.append(cur_hops)

        # calculate the energies
        with torch.no_grad():
            energy = model(x_cur).mean(axis=0)
            energies.append(energy.cpu().item())

        # plot the images
        wandb.log({"energy": energy.cpu().item(), 
                    "hops": cur_hops})
        if args.sampler in ["dmala", "acs"]: 
            a_s_mean = np.mean(sampler.a_s)
            wandb.log({"a_s": a_s_mean})
            acceptance_rates.append(a_s_mean)
            sampler.a_s = []


        if itr % args.viz_every == 0:
            img = wandb.Image(get_grid_img(x_init))
            wandb.log({f"sample_itr": img})
            # plot(f"{cur_dir}/step_{itr}.png", x_init)

        x_init = x_cur

    with open(f"{cur_dir}/times.pickle", "wb") as f:
        pickle.dump(times, f)
    with open(f"{cur_dir}/log_mmds.pickle", "wb") as f:
        pickle.dump(log_mmds, f)
    with open(f"{cur_dir}/hops.pickle", "wb") as f:
        pickle.dump(hops, f)
    with open(f"{cur_dir}/energies.pickle", "wb") as f:
        pickle.dump(energies, f)
    if args.sampler in ["acs", "dmala"]:
        with open(f"{cur_dir}/a_s.pickle", "wb") as f:
            pickle.dump(acceptance_rates, f)
    # with open(f"{cur_dir}/interp_energies.pickle", "wb") as f:
    #     pickle.dump(interp_energies, f)
    # if args.get_base_energies:
    #     with open(f"{cur_dir}/digit_energies.pickle", "wb") as f:
    #         pickle.dump(digit_energy_res, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default="raw_exp_data/ebm_sample")
    parser.add_argument("--n_samples", type=int, default=2)
    parser.add_argument("--n_test_samples", type=int, default=2)
    parser.add_argument("--seed_file", type=str, default="seed.txt")
    parser.add_argument("--ebm_model", type=str, default="resnet-64")
    # model def

    # logging
    parser.add_argument("--print_every", type=int, default=10)
    parser.add_argument("--viz_every", type=int, default=20)
    parser.add_argument("--cd", type=int, default=10)
    parser.add_argument("--img_size", type=int, default=28)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--samples_to_generate", type=int, default=128)
    parser.add_argument("--sampling_steps", type=int, default=5000)
    parser.add_argument("--dataset_name", type=str, default="dynamic_mnist")
    # for ess
    parser.add_argument("--model", type=str, default="resnet-64")
    parser.add_argument("--subsample", type=int, default=1)
    parser.add_argument("--burn_in", type=float, default=0.1)
    parser.add_argument(
        "--ess_statistic", type=str, default="dims", choices=["hamming", "dims"]
    )
    parser.add_argument("--no_ess", action="store_true")
    parser.add_argument("--test_batch_size", type=int, default=100)
    parser.add_argument("--scheduler_buffer_size", type=int, default=100)
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--base_dist", action="store_true")
    parser.add_argument("--sampler", type=str, default="acs")
    parser.add_argument("--cuda_id", type=int, default=0)
    # use mh in the begining of the sampling process during burning (first 500)
    # may have an impact on burn in and acceptance rate
    parser.add_argument("--initial_mh", action="store_true")
    parser.add_argument("--zero_init", action="store_true")
    parser.add_argument("--gt_sample_steps", type=int, default=10000)

    # adaptive arguments here
    parser = config_acs_args(parser)
    parser.add_argument("--step_size", type=float, default=.2)
    parser = config_acs_pcd_args(parser)
    parser.add_argument("--data", type=str, default="mnist")
    parser.add_argument("--get_base_energies", action="store_true")
    parser.add_argument("--use_acs_ebm", action="store_true")
    parser.add_argument("--use_diag_acs", action="store_true")
    parser.add_argument("--dula_burnin_steps", type=int, default=50)
    parser.add_argument("--norm_mterm", action="store_true")
    parser.add_argument("--acs_burnin_steps", type=int, default=0)
    parser.add_argument("--ebm_temp", type=float, default=1)
    parser.add_argument("--n_tiles", type=int, default=49)
    args = parser.parse_args()
    args.n_steps = args.sampling_steps
    main(args)
