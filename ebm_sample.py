import argparse
from result_storing_utils import *
import torch
import numpy as np
import vamp_utils
import os
import tensorflow_probability as tfp
import tqdm
import mlp
from pcd_ebm_ema import get_sampler, EBM
import torchvision
from rbm_sample import get_ess


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
    sampler_name = sampler.get_name()
    if args.burnin_adaptive:
        sampler_name += "_adapt_burnin"
        sampler_name += f"_lr_{args.burnin_lr}"
    cur_dir = f"{args.save_dir}/{sampler_name}"
    os.makedirs(cur_dir, exist_ok=True)

    # copying the same initialization for the buffer as in pcd_ebm_ema
    init_batch = []
    for x, _ in train_loader:
        init_batch.append(x.to(device))
    init_batch = torch.cat(init_batch, 0)
    eps = 1e-2
    init_mean = init_batch.mean(0) * (1.0 - 2 * eps) + eps
    init_dist = torch.distributions.Bernoulli(probs=init_mean)

    if args.base_dist:
        model = EBM(net, init_mean)
    else:
        model = EBM(net)

    d = torch.load(f"{args.ckpt_path}")
    model.load_state_dict(d["ema_model"])

    x_init = init_dist.sample((args.samples_to_generate,)).to(device)

    model = model.to(device)

    # TODO: add in measuring of energies for different digits from rbm sample

    # metrics to keep track of:
    energies = []
    hops = []
    sample_var = []
    if args.burnin_adaptive:
        x_init, burnin_res = sampler.run_adaptive_burnin(
            x_init.detach(), model, budget=500, steps_obj="alpha_max", lr=args.burnin_lr
        )
        with open(f"{cur_dir}/burnin_res.pickle", "wb") as f:
            pickle.dump(burnin_res, f)
    for itr in tqdm.tqdm(range(args.sampling_steps)):
        if "cyc" in args.sampler:
            x_cur = sampler.step(x_init, model, itr)
        else:
            x_cur = sampler.step(x_init, model)
        # calculate the hops

        h = (x_cur != x_init).float().view(x_init.size(0), -1).sum(-1).mean().item()
        hops.append(h)

        # calculate the energies
        with torch.no_grad():
            energy = model(x_cur).mean(axis=0)
            energies.append(energy.cpu().item())

        # plot the images
        if itr % args.viz_every == 0:
            plot(f"{cur_dir}/step_{itr}.png", x_init)
        x_init = x_cur

    with open(f"{cur_dir}/hops.pickle", "wb") as f:
        pickle.dump(hops, f)
    with open(f"{cur_dir}/energies.pickle", "wb") as f:
        pickle.dump(energies, f)
    if "dmala" in args.sampler:
        with open(f"{cur_dir}/a_s.pickle", "wb") as f:
            pickle.dump(sampler.a_s, f)
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default="./figs/ebm_sample")
    parser.add_argument("--n_steps", type=int, default=50000)
    parser.add_argument("--n_samples", type=int, default=2)
    parser.add_argument("--n_test_samples", type=int, default=2)
    parser.add_argument("--seed_file", type=str, default="seed.txt")
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="figs/ebm/best_ckpt_dynamic_mnist_dmala_stepsize_0.2_0.5_0.2.pt",
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
    parser.add_argument("--sampling_steps", type=int, default=500)
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
    parser.add_argument("--num_cycles", type=int, default=250)
    parser.add_argument("--step_size", type=float, default=2.0)
    parser.add_argument("--base_dist", action="store_true")
    parser.add_argument("--sampler", type=str, default="cyc_dmala")
    parser.add_argument("--initial_balancing_constant", type=float, default=1.0)
    parser.add_argument("--cuda_id", type=int, default=0)
    # adaptive arguments here
    parser.add_argument("--adaptive_cycles", type=int, default=150)
    parser.add_argument("--adapt_rate", type=float, default=0.025)
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
    parser.add_argument("--use_big", action="store_true")
    args = parser.parse_args()
    main(args)
