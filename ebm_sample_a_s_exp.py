import argparse
import samplers
import torch
import numpy as np
import vamp_utils
import os
import tensorflow_probability as tfp
import tqdm
import mlp
from samplers import LangevinSampler
from pcd_ebm_ema import get_sampler, EBM
import torchvision
from rbm_sample import get_ess
import mmd
from config_cmdline import config_acs_args, config_sampler_args, config_acs_pcd_args
import pickle
import time
import pandas as pd


def sqrt(x):
    int(torch.sqrt(torch.Tensor([x])))


def main(args):
    device = torch.device(
        "cuda:" + str(args.cuda_id) if torch.cuda.is_available() else "cpu"
    )
    args.device = device

    plot = lambda p, x: torchvision.utils.save_image(
        x.view(x.size(0), args.input_size[0], args.input_size[1], args.input_size[2]),
        p,
        normalize=True,
        nrow=int(x.size(0) ** 0.5),
    )

    def my_print(s):
        print(s)

    my_print("Loading data")
    train_loader, _, _, args = vamp_utils.load_dataset(args)

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

    # copying the same initialization for the buffer as in pcd_ebm_ema
    init_batch = []
    for x, _ in train_loader:
        init_batch.append(x.to(device))
    init_batch = torch.cat(init_batch, 0)
    eps = 1e-2
    init_mean = init_batch.mean(0) * (1.0 - 2 * eps) + eps
    init_var = init_batch.std(0) ** 2 * (1 - 4 * eps) + eps
    init_dist = torch.distributions.Bernoulli(probs=init_mean)

    if args.base_dist:
        model = EBM(net, init_mean)
    else:
        model = EBM(net)

    d = torch.load(f"{args.ckpt_path}")
    model.load_state_dict(d["ema_model"])
    x_init = init_dist.sample((args.samples_to_generate,)).to(device)
    model = model.to(device)

    sampler = samplers.LangevinSampler(
        784,
        n_steps=1,
        bal=0.5,
        fixed_proposal=False,
        approx=True,
        multi_hop=False,
        temp=1.0,
        step_size=0,
        mh=True,
    )
    sampler.mh = False
    sampler.step_size = 15
    sampler.bal = 0.97
    for _ in range(50):
        x_init = sampler.step(x_init, model).detach()
    sampler.step_size = 0.1
    sampler.mh = True
    step_increment = 0.1
    betas = [.8]
    res_step_mean = {"step_sizes": []}
    res_step_std = {"step_sizes": []}

    for b in betas:
        res_step_mean[b] = []
        res_step_std[b] = []

    os.makedirs(f"ebm_a_s_exp/{args.data}", exist_ok=True)
    while sampler.step_size <= 10.0:
        res_step_mean["step_sizes"].append(sampler.step_size)
        res_step_std["step_sizes"].append(sampler.step_size)
        for b in betas:
            sampler.bal = b
            x = x_init.clone()
            sampler.a_s = []

            for _ in tqdm.tqdm(
                range(args.sampling_steps), desc=f"bal: {b}, step: {sampler.step_size}"
            ):
                xhat = sampler.step(x.detach(), model).detach()
                x = xhat
            mean = np.mean(sampler.a_s)
            std = np.std(sampler.a_s)
            print(mean)
            res_step_mean[b].append(mean)
            res_step_std[b].append(mean)
        sampler.step_size += step_increment

    with open(f"ebm_a_s_exp/{args.data}/avg_a_s.pickle", "wb") as f:
        pickle.dump(res_step_mean, f)
    with open(f"ebm_a_s_exp/{args.data}/std_a_s.pickle", "wb") as f:
        pickle.dump(res_step_std, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default="raw_exp_res/ebm_stepsize_acceptance")
    parser.add_argument("--n_samples", type=int, default=2)
    parser.add_argument("--n_test_samples", type=int, default=2)
    parser.add_argument("--seed_file", type=str, default="seed.txt")
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="example_dmala_dynamic_mnist_ebm.pt",
    )
    parser.add_argument("--ebm_model", type=str, default="resnet-64")
    # model def

    # logging
    parser.add_argument("--print_every", type=int, default=10)
    parser.add_argument("--viz_every", type=int, default=20)

    # for rbm training

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
    parser.add_argument("--sampler", type=str, default="cyc_dmala")
    parser.add_argument("--cuda_id", type=int, default=0)
    # use mh in the begining of the sampling process during burning (first 500)
    # may have an impact on burn in and acceptance rate
    parser.add_argument("--initial_mh", action="store_true")
    parser.add_argument("--zero_init", action="store_true")
    parser.add_argument("--gt_sample_steps", type=int, default=10000)

    # adaptive arguments here
    parser = config_acs_args(parser)
    parser = config_sampler_args(parser)
    parser = config_acs_pcd_args(parser)
    parser.add_argument("--data", type=str, default="mnist")
    parser.add_argument("--get_base_energies", action="store_true")
    parser.add_argument("--mode_escape", action="store_true")
    args = parser.parse_args()
    args.n_steps = args.sampling_steps
    main(args)
