import argparse
import mlp
import torch
import numpy as np
from samplers import (
    PerDimGibbsSampler,
    MultiDiffSampler,
    DiffSampler,
    PerDimMetropolisSampler,
    DiffSamplerMultiDim,
    LangevinSampler,
)
import block_samplers
import torch.nn as nn
import os
import torchvision
from utils import get_dlp_samplers
import vamp_utils
import ais
import copy
import time
import pickle
import sys
from asbs_code.GBS.sampling.globally import (
    AnyscaleBalancedSampler,
    DiagAnyscaleBalancedSampler,
)
from config_cmdline import config_acs_args, config_acs_pcd_args


def makedirs(dirname):
    """
    Make directory only if it's not already there.
    """
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def get_sampler(args):
    data_dim = np.prod(args.input_size)
    if args.input_type == "binary":
        if args.sampler == "gibbs":
            sampler = PerDimGibbsSampler(data_dim, rand=False)
        elif args.sampler == "rand_gibbs":
            sampler = PerDimGibbsSampler(data_dim, rand=True)
        elif args.sampler.startswith("bg-"):
            block_size = int(args.sampler.split("-")[1])
            sampler = block_samplers.BlockGibbsSampler(data_dim, block_size)
        elif args.sampler.startswith("hb-"):
            block_size, hamming_dist = [int(v) for v in args.sampler.split("-")[1:]]
            sampler = block_samplers.HammingBallSampler(
                data_dim, block_size, hamming_dist
            )
        elif args.sampler == "gwg":
            sampler = DiffSampler(
                data_dim,
                1,
                fixed_proposal=False,
                approx=True,
                multi_hop=False,
                temp=2.0,
            )
        elif args.sampler.startswith("gwg-"):
            n_hops = int(args.sampler.split("-")[1])
            sampler = MultiDiffSampler(
                data_dim, 1, approx=True, temp=2.0, n_samples=n_hops
            )

        elif args.sampler == "asb":
            sampler = DiagAnyscaleBalancedSampler(args, alpha=0.6)
            # sampler = AnyscaleBalancedSampler(
            #     args, cur_type="1st", sigma=0.1, alpha=0.5, adaptive=1
            # )
            model_name = "anyscale"
        else:
            sampler = get_dlp_samplers(args.sampler, data_dim, args.device, args)
    else:
        if args.sampler == "gibbs":
            sampler = PerDimMetropolisSampler(data_dim, int(args.n_out), rand=False)
        elif args.sampler == "rand_gibbs":
            sampler = PerDimMetropolisSampler(data_dim, int(args.n_out), rand=True)
        elif args.sampler == "gwg":
            sampler = DiffSamplerMultiDim(data_dim, 1, approx=True, temp=2.0)
        else:
            raise ValueError("invalid sampler")
    return sampler


class EBM(nn.Module):
    def __init__(self, net, mean=None, temp = 1):
        super().__init__()
        self.net = net
        if mean is None:
            self.mean = None
        else:
            self.mean = nn.Parameter(mean, requires_grad=False)
        self.temp = temp

    def forward(self, x):
        if self.mean is None:
            bd = 0.0
        else:
            base_dist = torch.distributions.Bernoulli(probs=self.mean)
            bd = base_dist.log_prob(x).sum(-1)

        logp = self.net(x).squeeze()
        return (logp + bd) / self.temp 


def main(args):
    makedirs(args.save_dir)
    logger = open("{}/log.txt".format(args.save_dir), "w")

    def my_print(s):
        print(s)
        logger.write(str(s) + "\n")
        logger.flush()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # load data
    train_loader, val_loader, test_loader, args = vamp_utils.load_dataset(args)
    plot = lambda p, x: torchvision.utils.save_image(
        x.view(x.size(0), args.input_size[0], args.input_size[1], args.input_size[2]),
        p,
        normalize=True,
        nrow=int(x.size(0) ** 0.5),
    )

    def preprocess(data):
        if args.dynamic_binarization:
            return torch.bernoulli(data)
        else:
            return data

    # make model
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

    # get data mean and initialize buffer
    init_batch = []
    for x, _ in train_loader:
        init_batch.append(preprocess(x))
    init_batch = torch.cat(init_batch, 0)
    eps = 1e-2
    init_mean = init_batch.mean(0) * (1.0 - 2 * eps) + eps
    if args.buffer_init == "mean":
        if args.input_type == "binary":
            init_dist = torch.distributions.Bernoulli(probs=init_mean)
            buffer = init_dist.sample((args.buffer_size,))
        else:
            buffer = None
            raise ValueError("Other types of data not yet implemented")

    elif args.buffer_init == "data":
        all_inds = list(range(init_batch.size(0)))
        init_inds = np.random.choice(all_inds, args.buffer_size)
        buffer = init_batch[init_inds]
    elif args.buffer_init == "uniform":
        buffer = (
            torch.ones(args.buffer_size, *init_batch.size()[1:]) * 0.5
        ).bernoulli()
    else:
        raise ValueError("Invalid init")

    if args.base_dist:
        model = EBM(net, init_mean)
    else:
        model = EBM(net)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    ema_model = copy.deepcopy(model)
    eval_sampler = LangevinSampler(
        np.prod(args.input_size),
        n_steps=1,
        bal=0.5,
        fixed_proposal=False,
        approx=True,
        multi_hop=False,
        temp=1.0,
        step_size=0.2,
        mh=True,
    )

    if args.ckpt_path is not None:
        d = torch.load(args.ckpt_path)
        model.load_state_dict(d["model"])
        ema_model.load_state_dict(d["ema_model"])
        # optimizer.load_state_dict(d['optimizer'])
        buffer = d["buffer"]

    # move to cuda
    model.to(device)
    ema_model.to(device)

    # get sampler
    sampler = get_sampler(args)

    my_print(device)
    my_print(model)
    my_print(buffer.size())
    my_print(sampler)

    itr = 0
    best_val_ll = -np.inf
    hop_dists = []
    all_inds = list(range(args.buffer_size))
    lr = args.lr
    init_dist = torch.distributions.Bernoulli(probs=init_mean.to(device))
    reinit_dist = torch.distributions.Bernoulli(probs=torch.tensor(args.reinit_freq))
    test_ll_list = []
    alpha_max_metrics_running = []
    alpha_min_metrics_running = []
    while itr < args.n_iters:
        for x in train_loader:
            if itr < args.warmup_iters:
                lr = args.lr * float(itr) / args.warmup_iters
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr

            x = preprocess(x[0].to(device).requires_grad_())

            # choose random inds from buffer
            buffer_inds = sorted(
                np.random.choice(all_inds, args.batch_size, replace=False)
            )
            x_buffer = buffer[buffer_inds].to(device)
            reinit = reinit_dist.sample((args.batch_size,)).to(device)
            x_reinit = init_dist.sample((args.batch_size,)).to(device)
            x_fake = x_reinit * reinit[:, None] + x_buffer * (1.0 - reinit[:, None])

            hops = []  # keep track of how much the sampelr moves particles around
            st = time.time()
            sampling_steps = args.sampling_steps
            if args.use_manual_EE:
                cycle_num = itr // args.steps_per_cycle
                step_num = itr % args.steps_per_cycle
                if args.burnin_adaptive:
                    if (
                        args.use_manual_EE
                        and step_num == 0
                        and cycle_num % args.adapt_every == 0
                    ):
                        (
                            x_fake,
                            alpha_max,
                            alpha_max_metrics,
                        ) = sampler.adapt_big_step(
                            x_init=x_fake.detach(),
                            model=model,
                            budget=args.burnin_budget,
                            init_big_step=5,
                            init_big_bal=0.8,
                            lr=args.burnin_lr,
                            test_steps=1,
                            a_s_cut=args.a_s_cut,
                            use_dula=True,
                            bdmala=None,
                        )
                        alpha_max_metrics_running.append(alpha_max_metrics)
                        print(alpha_max)
                    elif (
                        args.use_manual_EE
                        and step_num == 1
                        and cycle_num % args.adapt_every == 0
                    ):
                        (
                            x_fake,
                            alpha_min,
                            alpha_min_metrics,
                        ) = sampler.adapt_small_step(
                            x_init=x_fake,
                            model=model,
                            budget=args.burnin_budget,
                            init_small_step=0.05,
                            init_small_bal=0.5,
                            lr=args.burnin_lr,
                            test_steps=1,
                            a_s_cut=args.a_s_cut,
                            use_dula=False,
                            bdmala=None,
                        )
                        alpha_min_metrics_running.append(alpha_min_metrics)
                        print(alpha_min)
                    else:
                        if step_num == 0:
                            sampler.mh = False
                            sampling_steps = args.big_step_sampling_steps
                        else:
                            sampler.mh = True
                            sampling_steps = args.sampling_steps

                        for k in range(sampling_steps):
                            x_fake_new = sampler.step(x_fake.detach(), model, itr).detach()
                            x_fake = x_fake_new
            else:
                for k in range(args.sampling_steps):
                    x_fake_new = sampler.step(x_fake.detach(), model).detach()
                    x_fake = x_fake_new
            st = time.time() - st
            hop_dists.append(np.mean(hops))

            # update buffer
            buffer[buffer_inds] = x_fake.detach().cpu()

            logp_real = model(x).squeeze()
            if args.p_control > 0:
                grad_ld = (
                    torch.autograd.grad(logp_real.sum(), x, create_graph=True)[0]
                    .flatten(start_dim=1)
                    .norm(2, 1)
                )
                grad_reg = (grad_ld**2.0 / 2.0).mean() * args.p_control
            else:
                grad_reg = 0.0

            logp_fake = model(x_fake).squeeze()

            obj = logp_real.mean() - logp_fake.mean()
            loss = (
                -obj
                + grad_reg
                + args.l2 * ((logp_real**2.).mean() + (logp_fake**2.).mean())
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update ema_model
            for p, ema_p in zip(model.parameters(), ema_model.parameters()):
                ema_p.data = ema_p.data * args.ema + p.data * (1.0 - args.ema)

            if itr % args.print_every == 0:
                my_print(
                    "({}) | ({}/iter) cur lr = {:.4f} |log p(real) = {:.4f}, "
                    "log p(fake) = {:.4f}, diff = {:.4f}, hops = {:.4f}".format(
                        itr,
                        st,
                        lr,
                        logp_real.mean().item(),
                        logp_fake.mean().item(),
                        obj.item(),
                        hop_dists[-1],
                    )
                )
            if itr % args.viz_every == 0:
                plot("{}/data_{}.png".format(args.save_dir, itr), x.detach().cpu())
                plot("{}/buffer_{}.png".format(args.save_dir, itr), x_fake)

            if (itr + 1) % args.eval_every == 0:
                logZ, train_ll, val_ll, test_ll, ais_samples = ais.evaluate(
                    ema_model,
                    init_dist,
                    eval_sampler,
                    train_loader,
                    val_loader,
                    test_loader,
                    preprocess,
                    device,
                    args.eval_sampling_steps,
                    args.test_batch_size,
                )
                my_print(
                    "EMA Train log-likelihood ({}): {}".format(itr, train_ll.item())
                )
                my_print("EMA Valid log-likelihood ({}): {}".format(itr, val_ll.item()))
                my_print("EMA Test log-likelihood ({}): {}".format(itr, test_ll.item()))
                test_ll_list.append(test_ll.item())
                for _i, _x in enumerate(ais_samples):
                    plot(
                        "{}/EMA_sample_{}_{}_{}_{}_{}.png".format(
                            args.save_dir,
                            args.dataset_name,
                            args.sampler,
                            args.step_size,
                            itr,
                            _i,
                        ),
                        _x,
                    )

                model.cpu()
                d = {}
                d["model"] = model.state_dict()
                d["ema_model"] = ema_model.state_dict()
                d["buffer"] = buffer
                d["optimizer"] = optimizer.state_dict()
                if val_ll.item() > 0:
                    exit()
                if val_ll.item() > best_val_ll:
                    best_val_ll = val_ll.item()
                    my_print("Best valid likelihood")
                    torch.save(
                        d,
                        "{}/best_ckpt_{}_{}_{}.pt".format(
                            args.save_dir,
                            args.dataset_name,
                            args.sampler,
                            args.step_size,
                        ),
                    )
                else:
                    torch.save(
                        d,
                        "{}/ckpt_{}_{}_{}.pt".format(
                            args.save_dir,
                            args.dataset_name,
                            args.sampler,
                            args.step_size,
                        ),
                    )

                model.to(device)

            itr += 1
    np.save(
        "{}/test_ll_{}_{}_{}.npy".format(
            args.save_dir, args.dataset_name, args.sampler, args.step_size
        ),
        test_ll_list,
    )
    if args.burnin_adaptive:
        pickle.dump(
            alpha_max_metrics_running,
            open(f"{args.save_dir}/alpha_max_metrics.pickle", "wb"),
        )
        pickle.dump(
            alpha_min_metrics_running,
            open(f"{args.save_dir}/alpha_min_metrics.pickle", "wb"),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument("--save_dir", type=str, default="./figs/ebm")
    parser.add_argument("--dataset_name", type=str, default="static_mnist")
    parser.add_argument("--ckpt_path", type=str, default=None)
    # data generation
    parser.add_argument("--n_out", type=int, default=3)  # potts
    # models
    parser.add_argument("--model", type=str, default="mlp-256")
    parser.add_argument("--base_dist", action="store_true")
    parser.add_argument("--p_control", type=float, default=0.0)
    parser.add_argument("--l2", type=float, default=0.0)
    parser.add_argument("--ema", type=float, default=0.999)
    # mcmc
    parser.add_argument("--sampler", type=str, default="gibbs")
    parser.add_argument("--seed", type=int, default=1234567)
    parser.add_argument("--sampling_steps", type=int, default=100)
    parser.add_argument("--reinit_freq", type=float, default=0.0)
    parser.add_argument("--eval_sampling_steps", type=int, default=100)
    parser.add_argument("--buffer_size", type=int, default=1000)
    parser.add_argument("--buffer_init", type=str, default="mean")
    # training
    parser.add_argument("--steps_per_cycle", type=int, default=100)
    parser.add_argument("--n_iters", type=int, default=100000)
    parser.add_argument("--warmup_iters", type=int, default=-1)
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--test_batch_size", type=int, default=100)
    parser.add_argument("--print_every", type=int, default=100)
    parser.add_argument("--viz_every", type=int, default=1000)
    parser.add_argument("--eval_every", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--cuda_id", type=int, default=0)
    parser.add_argument("--step_size", type=float, default=.15)
    parser = config_acs_pcd_args(parser)
    # sbc hyper params
    parser = config_acs_args(parser)
    parser = config_acs_pcd_args(parser)
    args = parser.parse_args()
    args.num_cycles = args.n_iters // args.steps_per_cycle
    device = torch.device(
        "cuda:" + str(args.cuda_id) if torch.cuda.is_available() else "cpu"
    )
    args.device = device
    args.n_steps = args.n_iters
    main(args)
