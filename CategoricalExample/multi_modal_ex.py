from re import MULTILINE
import torch.nn as nn
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from torch.distributions import Binomial
from torch import tensor
from samplers import LangevinSamplerOrdinal
from tqdm import tqdm

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


class SimpleBin(nn.Module):
    def __init__(self, ss_dim, mean, device) -> None:
        super().__init__()
        self.ss_dim = ss_dim
        self.mean = mean
        self.x_prob = self.mean[0] / self.ss_dim
        self.y_prob = self.mean[1] / self.ss_dim
        self.x_rv = Binomial(
            total_count=tensor([self.ss_dim]).to(device),
            probs=tensor([self.x_prob]).to(device),
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

a   def forward(self, x):
        out = torch.zeros((x.shape[0])).to(self.device)
        for bv in self.bvs:
            res = bv(x).exp() * (1 / len(self.bvs))
            out += res
        return torch.log(out + EPS)


class MM_Heat(nn.Module):
    def __init__(self, ss_dim, means, var, device) -> None:
        super().__init__()
        self.means = means.float()
        self.means
        self.ss_dim = ss_dim
        self.one_hot = False
        self.var = var
        self.device = device

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


def run_sampler(
    sampler,
    energy_function,
    sampling_steps,
    burnin_steps,
    batch_size,
    device,
    start_coord,
):
    energy_function.one_hot = False
    # x = torch.zeros((batch_size, 2)).to(device)
    x = torch.full((batch_size, 2), fill_value=start_coord).to(device)
    samples = []
    burnin_a_s = []
    chain_a_s = []
    pg = tqdm(range(sampling_steps + burnin_steps))
    for i in pg:
        x = x.to(device)
        energy_function = energy_function.to(device)
        x = sampler.step(x.long().detach(), energy_function).detach()
        # storing the acceptance rate
        if i < burnin_steps:
            burnin_a_s.append(sampler.a_s)
        else:
            chain_a_s.append(sampler.a_s)
            samples += list(x.long().detach().cpu().numpy())
        # resetting the acceptance rate
        sampler.a_s = []

    return burnin_a_s, chain_a_s, samples


def get_sampler(args):
    if args.sampler == "dmala":
        sampler = LangevinSamplerOrdinal(
            dim=int(2), max_val=int(DIM), n_steps=1, mh=True, step_size=args.step_size
        )
    elif args.sampler == "dula":
        sampler = LangevinSamplerOrdinal(
            dim=int(2), max_val=int(DIM), n_steps=1, mh=False, step_size=args.step_size
        )
    return sampler


def main(args):
    device = torch.device(
        "cuda:" + str(args.cuda_id) if torch.cuda.is_available() else "cpu"
    )
    if args.modality == "unimodal":
        modes_to_use = UNIMODAL
    else:
        modes_to_use = MULTIMODAL
    if args.dist_type == "heat":
        energy_function = MM_Heat(
            ss_dim=DIM, means=modes_to_use, var=args.dist_var, device=device
        )
    elif args.dist_type == "bin":
        energy_function = MM_Bin(ss_dim=DIM, means=modes_to_use, device=device)

    cur_dir = f"{args.save_dir}/{args.dist_type}_{args.dist_var}/"
    cur_dir += f"{args.modality}_{args.starting_point}/"
    cur_dir += f"{args.sampler}_{args.initial_balancing_constant}_{args.step_size}/"
    os.makedirs(cur_dir, exist_ok=True)
    # plotting the ground truth distribution
    samples, probs = calc_probs(DIM, energy_function, device)
    dist_img = np.zeros((DIM, DIM))
    for i in range(len(samples)):
        coord = samples[i]
        dist_img[coord[0], coord[1]] = probs[i]
    pickle.dump(tensor(dist_img), open(f"{cur_dir}/gt.pickle", "wb"))
    plt.imshow(dist_img)

    plt.savefig(f"{cur_dir}/init_dist.png")
    if args.starting_point == "center":
        start_coord = 32
    elif args.starting_point == "low_mode":
        # TODO: for bin, make it start at the LOWEST mode
        start_coord = 50
    else:
        start_coord = np.random.randint(0, 64)
    est_img = torch.zeros((DIM, DIM))
    sampler = get_sampler(args)
    burnin_a_s, chain_a_s, samples = run_sampler(
        energy_function=energy_function,
        burnin_steps=args.burnin_steps,
        batch_size=args.batch_size,
        sampling_steps=args.sampling_steps,
        device=device,
        sampler=sampler,
        start_coord=start_coord,
    )
    for i in range(len(samples)):
        coord = samples[i]
        est_img[coord[0], coord[1]] += 1
    pickle.dump(burnin_a_s, open(f"{cur_dir}/burnin_a_s.pickle", "wb"))
    pickle.dump(chain_a_s, open(f"{cur_dir}/chain_a_s.pickle", "wb"))
    pickle.dump(est_img, open(f"{cur_dir}/actual_probs.pickle", "wb"))
    plt.imshow(est_img)
    plt.savefig(f"{cur_dir}/est_dist.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dist_type", type=str, default="heat")
    parser.add_argument("--save_dir", type=str, default="./figs/multi_modal")
    parser.add_argument("--cuda_id", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--burnin_steps", type=int, default=2000)
    parser.add_argument("--sampling_steps", type=int, default=5000)
    parser.add_argument("--step_size", type=float, default=0.1)
    parser.add_argument("--dist_var", type=float, default=10)
    parser.add_argument("--initial_balancing_constant", type=float, default=0.5)
    parser.add_argument("--sampler", type=str, default="dmala")
    parser.add_argument("--modality", type=str, default="multimodal")
    parser.add_argument("--starting_point", type=str, default="center")
    args = parser.parse_args()
    main(args)
