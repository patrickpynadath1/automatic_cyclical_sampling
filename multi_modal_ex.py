from config_cmdline import config_SbC_args, config_sampler_args, config_adaptive_args
import torch.nn as nn
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from itertools import product
import argparse
from samplers import PerDimGibbsSamplerOrd
from torch.distributions import Binomial
from torch import tensor
from samplers import LangevinSamplerOrdinal, CyclicalLangevinSamplerOrdinal
from tqdm import tqdm

from asbs_code.GBS.sampling.globally import AnyscaleBalancedSampler

# I dont know why, but this magically solves all numerical instability problems
EPS = 1e-10

# some hardcoded examples;


DIM = 64
mean1 = [10, 50]
mean2 = [50, 50]
mean3 = [10, 10]
mean4 = [50, 10]
center_mean = [32, 32]
torch.cuda.manual_seed_all(0)
torch.manual_seed(0)
MULTIMODAL = torch.tensor([mean1, mean2, mean3, mean4])
UNIMODAL = tensor([center_mean])


def get_modes(num_modes, space_between_modes=5):
    # what should the mode centers be?
    dim = (num_modes + 1) * space_between_modes
    mode_centers_x = np.linspace(0, dim, num_modes + 2)[1:-1]
    mode_centers = list(product(mode_centers_x, repeat=2))
    return dim, torch.Tensor(mode_centers)


class SimpleBin(nn.Module):
    def __init__(self, ss_dim, mean, device) -> None:
        super().__init__()
        self.ss_dim = ss_dim
        self.mean = mean
        self.x_prob = self.mean[0] / self.ss_dim
        self.y_prob = self.mean[1] / self.ss_dim
        self.x_rv = Binomial(
            total_count=tensor([self.ss_dim]).to(device),
            probs=tensor([self.x_prob]).td(device),
        )
        self.y_rv = Binomial(
            total_count=tensor([self.ss_dim]).to(device),
            probs=tensor([self.y_prob]).to(device),
        )

    def forward(self, x):
        return self.x_rv.log_prob(x[:, 0]) + self.y_rv.log_prob(x[:, 1])


class MM_Bin(nn.Module):
    def __init__(self, ss_dim, device, means) -> None:
        super().__init__()
        self.ss_dim = ss_dim
        self.means = means
        self.bvs = []
        self.device = device
        for i in range(self.means.shape[0]):
            bv = SimpleBin(self.ss_dim, self.means[i, :], device=self.device)
            self.bvs.append(bv)

    def forward(self, x):
        out = torch.zeros((x.shape[0])).to(self.device)
        for bv in self.bvs:
            res = bv(x).exp() * (1 / len(self.bvs))
            out += res
        return torch.log(out + EPS)


class MM_Heat(nn.Module):
    def __init__(self, ss_dim, means, var, device, weights=None) -> None:
        super().__init__()
        self.means = means.float()
        self.means
        self.ss_dim = ss_dim
        self.one_hot = False
        self.var = var
        self.device = device
        self.weights = weights
        self.L = 1

    def forward(self, x):
        x = x.float()
        out = torch.zeros((x.shape[0])).to(x.device)
        self.means = self.means.to(x.device)
        if self.one_hot:
            dim_v = tensor(
                [[i for i in range(self.ss_dim)], [i for i in range(self.ss_dim)]]
            ).to(x.device)
            # turning from 1 hot to coordinate vector (2 dim with ss_dim potenital values)
            x = (dim_v * x).sum(axis=2)
        for m in range(self.means.shape[0]):
            if self.weights:
                out += (
                    torch.exp(
                        (-torch.norm(x - self.means[m, :], dim=1))
                        * (1 / (self.var * self.means.shape[0]))
                    )
                    * self.weights[m]
                )
            else:
                out += torch.exp(
                    (-torch.norm(x - self.means[m, :], dim=1))
                    * (1 / (self.var * self.means.shape[0]))
                )
            # for i in range(x.shape[0]):
            #     out[i] += torch.exp(
            #         (-torch.norm(x[i, :] - self.means[m, :]))
            #         * 1
            #         / (self.var * self.means.shape[0])
            #     )
        return torch.log(out + EPS)


def calc_probs(ss_dim, energy_function, device):
    energy_function.one_hot = False
    samples = []
    for i in range(ss_dim):
        for j in range(ss_dim):
            samples.append((i, j))
    samples = tensor(samples).to(device)
    energies = energy_function(samples)
    z = energies.exp().sum()
    probs = energies.exp() / z
    return samples, probs


def run_sampler_burnin(sampler, energy_function, start_coord, device, args):
    x = torch.Tensor(start_coord).repeat(args.batch_size, 1).to(device)
    # TODO: fix the burnin adaptive; make sure it works for thiis
    _, burnin_res = sampler.adapt_alg_greedy_mod(
        x.detach(),
        energy_function,
        budget=args.burnin_budget,
        test_steps=1,
        init_big_step=200,  # TODO: make not hard coded
        init_small_step=2,
        init_big_bal=0.6,
        init_small_bal=0.5,
        lr=args.burnin_lr,
        a_s_cut=args.a_s_cut,
        bal_resolution=args.bal_resolution,
    )
    return burnin_res


def run_sampler(
    sampler,
    energy_function,
    sampling_steps,
    batch_size,
    device,
    start_coord,
    is_cyc,
    dim,
    x_init=None,
    show_a_s=True,
    rand_restarts=10,
):
    energy_function.one_hot = False
    # x = torch.zeros((batch_size, 2)).to(device)
    if x_init is not None:
        x = x_init
    else:
        x = torch.Tensor(start_coord).repeat(batch_size, 1).to(device)
    samples = []
    chain_a_s = []
    pg = tqdm(range(sampling_steps))
    restart_every = sampling_steps // rand_restarts
    for i in pg:
        x = x.to(device)
        energy_function = energy_function.to(device)
        if is_cyc:
            if i % sampler.iter_per_cycle == 0:
                sampler.mh = True
            else:
                sampler.mh = True
            x = sampler.step(x.long().detach(), energy_function, i).detach()
        else:
            x = sampler.step(x.long().detach(), energy_function).detach()
        # storing the acceptance rate
        samples += list(x.long().detach().cpu().numpy())
        if show_a_s:
            chain_a_s.append(sampler.a_s)
            samples += list(x.long().detach().cpu().numpy())
            # resetting the acceptance rate
            sampler.a_s = []
            if i % 10 == 0:
                pg.set_description(
                    f"mean a_s: {np.mean(chain_a_s[-100:])}, step_size: {sampler.step_size}"
                )
        # if i % restart_every == 0:
        #     x = torch.randint(0, dim, (1, 2)).repeat((batch_size, 1)).to(device)
    return chain_a_s, samples


def get_sampler(args, dim, device):
    use_mh = "dmala" in args.sampler
    if "cyc" in args.sampler:
        sampler = CyclicalLangevinSamplerOrdinal(
            dim=int(2),
            max_val=dim,
            n_steps=1,
            mh=use_mh,
            num_cycles=args.num_cycles,
            num_iters=args.sampling_steps,
            mean_stepsize=args.step_size,
            initial_balancing_constant=args.initial_balancing_constant,
            burnin_adaptive=args.burnin_adaptive,
            burnin_budget=args.burnin_budget,
            burnin_lr=args.burnin_lr,
            sbc=args.use_manual_EE,
            big_step=args.big_step,
            big_bal=args.big_bal,
            small_step=args.small_step,
            small_bal=args.small_bal,
            min_lr=args.min_lr,
            device=device,
        )
        if args.hand_tuned_sched:
            step_sizes = torch.Tensor([150, 75, 75, 75, 75, 10, 10, 5]).to(device)
            bal = torch.Tensor([0.5, 0.5, 0.5, 0.5, 0.5, 0.5]).to(device)
            sampler.step_sizes = step_sizes
            sampler.balancing_constants = bal
            sampler.iter_per_cycle = len(step_sizes)

        print(sampler.step_sizes)
        print(sampler.balancing_constants)
    elif args.sampler == "dmala":
        sampler = LangevinSamplerOrdinal(
            dim=int(2),
            max_val=dim,
            n_steps=1,
            mh=use_mh,
            step_size=args.step_size,
            bal=args.initial_balancing_constant,
            device=device,
        )
    elif args.sampler == "gibbs":
        sampler = PerDimGibbsSamplerOrd(dim=2, max_val=dim)
    elif args.sampler == "asb":
        sampler = AnyscaleBalancedSampler(
            args, cur_type="1st", sigma=0.1, alpha=0.5, adaptive=1
        )
    return sampler


def main(args):
    device = torch.device(
        "cuda:" + str(args.cuda_id) if torch.cuda.is_available() else "cpu"
    )

    # dim, modes_to_use = get_modes(args.num_modes, args.space_between_modes)
    if args.modality == "unimodal":
        modes_to_use = UNIMODAL
    else:
        modes_to_use = MULTIMODAL
    dim = DIM
    if args.dist_type == "heat":
        energy_function = MM_Heat(
            ss_dim=dim,
            means=modes_to_use,
            var=args.dist_var,
            device=device,
        )
    elif args.dist_type == "bin":
        energy_function = MM_Bin(ss_dim=DIM, means=modes_to_use, device=device)
    else:
        energy_function = None
    sampler = get_sampler(args, dim, device)
    cur_dir = f"{args.save_dir}/{args.dist_type}_{args.dist_var}/"
    cur_dir += f"{args.modality}_{args.starting_point}/"
    if "cyc" in args.sampler:
        cur_dir += f"{sampler.get_name()}/"
    elif args.sampler == "dmala" or args.sampler == "dula":
        cur_dir += f"{args.sampler}_{args.initial_balancing_constant}_{args.step_size}/"
    else:
        cur_dir += f"{args.sampler}"
    os.makedirs(cur_dir, exist_ok=True)
    # plotting the ground truth distribution
    samples, probs = calc_probs(dim, energy_function, device)
    dist_img = np.zeros((dim, dim))
    for i in range(len(samples)):
        coord = samples[i]
        dist_img[coord[0], coord[1]] = probs[i]

    pickle.dump(tensor(dist_img), open(f"{cur_dir}/gt.pickle", "wb"))
    plt.imshow(dist_img)

    plt.axis("off")
    plt.savefig(f"{cur_dir}/init_dist.svg")
    plt.savefig(f"{cur_dir}/init_dist.png")
    print(f"gt: {cur_dir}/init_dist.png")
    # if args.starting_point == "center":
    #     start_coord = 32
    # elif args.starting_point == "low_mode":
    #     # TODO: for bin, make it start at the LOWEST mode
    #     start_coord = 63
    # else:
    start_coord_y = np.random.randint(0, dim)
    start_coord_x = np.random.randint(0, dim)
    # start_coord_x = start_coord_y = dim // 4
    start_coord = (start_coord_x, start_coord_y)
    est_img = torch.zeros((dim, dim))
    if args.burnin_adaptive:
        burnin_res = run_sampler_burnin(
            sampler=sampler,
            device=device,
            start_coord=start_coord,
            energy_function=energy_function,
            args=args,
        )
        pickle.dump(burnin_res, open(f"{cur_dir}/burnin_res.pickle", "wb"))
    x_init = None
    if args.sampler in ["gibbs", "asb", "dula", "cyc_dula"]:
        show_a_s = False
    else:
        show_a_s = True
    chain_a_s, samples = run_sampler(
        energy_function=energy_function,
        batch_size=args.batch_size,
        sampling_steps=args.sampling_steps,
        device=device,
        sampler=sampler,
        start_coord=start_coord,
        is_cyc="cyc" in args.sampler,
        x_init=x_init,
        show_a_s=show_a_s,
        dim=dim,
    )
    for i in range(len(samples)):
        coord = samples[i]
        est_img[coord[0], coord[1]] += 1
    pickle.dump(chain_a_s, open(f"{cur_dir}/chain_a_s.pickle", "wb"))
    pickle.dump(est_img, open(f"{cur_dir}/actual_probs.pickle", "wb"))
    plt.imshow(est_img)
    plt.axis("off")
    plt.savefig(f"{cur_dir}/est_dist.png")
    plt.savefig(f"{cur_dir}/est_dist.svg")
    print(f"est: {cur_dir}/est_dist.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dist_type", type=str, default="heat")
    parser.add_argument("--save_dir", type=str, default="./figs/multi_modal")
    parser.add_argument("--cuda_id", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--sampling_steps", type=int, default=1000)
    parser.add_argument("--dist_var", type=float, default=0.9)
    parser.add_argument("--sampler", type=str, default="dmala")
    parser.add_argument("--modality", type=str, default="multimodal")
    parser.add_argument("--starting_point", type=str, default="center")
    parser.add_argument("--num_modes", type=int, default=5)
    parser.add_argument("--space_between_modes", type=int, default=50)
    parser.add_argument("--scheduler_buffer_size", type=int, default=100)
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--hand_tuned_sched", action="store_true")
    parser = config_sampler_args(parser)
    parser = config_adaptive_args(parser)
    parser = config_SbC_args(parser)
    args = parser.parse_args()
    args.burn_in = 0
    args.n_steps = 1
    main(args)
