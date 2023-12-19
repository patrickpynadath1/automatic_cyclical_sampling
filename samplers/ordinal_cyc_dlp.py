import math
import torch
import torch.nn as nn
import utils
import numpy as np
from .adaptive_components import *
from .ordinal_dlp import LangevinSamplerOrdinal


class CyclicalLangevinSamplerOrdinal(nn.Module):
    def __init__(
        self,
        dim,
        device,
        max_val,
        num_cycles,
        num_iters,
        n_steps=10,
        approx=False,
        multi_hop=False,
        fixed_proposal=False,
        temp=1.0,
        mean_stepsize=0.2,
        mh=True,
        initial_balancing_constant=1,
        burnin_adaptive=False,
        burnin_budget=500,
        burnin_lr=0.5,
        adapt_alg="twostage_optim",
        sbc=False,
        big_step=None,
        big_bal=None,
        small_step=None,
        small_bal=None,
        iter_per_cycle=None,
        min_lr=False,
    ):
        super().__init__()
        self.device = device
        self.dim = dim
        self.max_val = max_val
        self.n_steps = n_steps
        self._ar = 0.0
        self._mt = 0.0
        self._pt = 0.0
        self._hops = 0.0
        self._phops = 0.0
        self.approx = approx
        self.fixed_proposal = fixed_proposal
        self.multi_hop = multi_hop
        self.temp = temp
        self.step_size = mean_stepsize
        self.initial_step_size = mean_stepsize
        self.num_cycles = num_cycles
        self.num_iters = num_iters
        self.burnin_adaptive = burnin_adaptive
        self.burnin_budget = burnin_budget
        self.burnin_lr = burnin_lr
        self.mh = mh
        self.a_s = []
        self.hops = []
        self.initial_balancing_constant = initial_balancing_constant
        self.balancing_constant = initial_balancing_constant
        self.adapt_alg = adapt_alg
        if iter_per_cycle and sbc:
            self.iter_per_cycle = iter_per_cycle
        else:
            self.iter_per_cycle = math.ceil(self.num_iters / self.num_cycles)
        self.step_sizes = self.calc_stepsizes(mean_stepsize)
        self.diff_values = []
        self.flip_probs = []
        self.balancing_constants = self.calc_balancing_constants(
            self.initial_balancing_constant
        )
        if sbc and big_step and big_bal and small_step and small_bal:
            self.sbc = sbc
            self.big_step = big_step
            self.small_step = small_step
            self.big_bal = big_bal
            self.small_bal = small_bal
            self.step_sizes = [big_step] + [small_step] * (self.iter_per_cycle - 1)
            self.balancing_constants = [big_bal] + [small_bal] * (
                self.iter_per_cycle - 1
            )

        else:
            self.sbc = False
        self.min_lr = min_lr
        if self.min_lr:
            self.min_lr_cutoff()

    def get_name(self):
        if self.mh:
            base = "cyc_dmala"
        else:
            base = "cyc_dula"
        if self.burnin_adaptive:
            name = f"{base}_cycles_{self.num_cycles}"
            name = (
                name
                + f"_{self.adapt_alg}_budget_{self.burnin_budget}_lr_{self.burnin_lr}"
            )
        else:
            name = f"{base}_cycles_{self.num_cycles}"
            name = (
                name
                + f"_stepsize_{self.initial_step_size}_initbal_{self.balancing_constant}"
            )
        if self.sbc:
            name = (
                base
                + f"_cycle_length_{self.iter_per_cycle}_big_s_{self.big_step}_b_{self.big_bal}_small_s_{self.small_step}_b_{self.small_bal}"
            )
        if self.min_lr:
            name += "_min_lr"
        return name

    def min_lr_cutoff(self):
        if self.mh:
            min_step = 0.2
        else:
            min_step = 0.1
        for i in range(len(self.step_sizes)):
            if self.step_sizes[i] <= min_step:
                self.step_sizes[i] = min_step
                self.balancing_constants[i] = 0.5

    def calc_stepsizes(self, mean_step):
        res = []
        total_iter = self.iter_per_cycle
        for k_iter in range(total_iter):
            inner = (np.pi * k_iter) / total_iter
            cur_step_size = mean_step * (np.cos(inner) + 1)
            step_size = cur_step_size
            res.append(step_size)
        res = torch.tensor(res, device=self.device)
        return res

    def calc_opt_acc_rate(self):
        res = []
        total_iter = self.iter_per_cycle
        a_0 = 0.148
        for k_iter in range(total_iter):
            inner = (np.pi * k_iter) / total_iter
            cur_acc = (a_0 - 1) / 2 * (np.cos(-inner)) + (a_0 + 1) / 2
            res.append(cur_acc)
        res = torch.tensor(res, device=self.device)
        return res

    def calc_balancing_constants(self, init_bal):
        res = []
        total_iter = self.iter_per_cycle
        for k_iter in range(total_iter):
            inner = (np.pi * k_iter) / total_iter
            cur_balancing_constant = (init_bal - 0.5) / 2 * (np.cos(inner)) + (
                init_bal + 0.5
            ) / 2
            res.append(cur_balancing_constant)
        res = torch.tensor(res, device=self.device)
        return res

    def adapt_steps_sl_w(
        self,
        x_init,
        model,
        budget,
        init_step_size,
        init_bal,
        window_range=0.1,
        resolution=3,
        test_steps=10,
        a_s_cut=0.6,
        lr=0.5,
        error_margin_alphamax=0.01,
        tune_stepsize=True,
    ):
        bdmala = LangevinSamplerOrdinal(
            dim=self.dim,
            n_steps=self.n_steps,
            approx=self.approx,
            multi_hop=self.multi_hop,
            fixed_proposal=self.fixed_proposal,
            step_size=1,
            mh=True,
            bal=init_bal,
        )
        steps_to_test = np.linspace(
            max(init_step_size - window_range),
            init_step_size + window_range,
            resolution,
        )

        def step_update(sampler, alpha):
            sampler.step_size = alpha

        a_s_l, hops_l, x_potential_l = run_hyperparameters(
            model, bdmala, x_init, test_steps, step_update, steps_to_test
        )
        a_s_l_close = [np.abs(a - a_s_cut) for a in a_s_l]
        cur_step, a_s, hops, x_cur = update_hyperparam_metrics(
            np.argmin(a_s_l_close), steps_to_test, a_s_l, hops_l, x_potential_l
        )
        return x_cur, cur_step, a_s

    def adapt_steps(
        self,
        x_init,
        model,
        budget,
        test_steps=10,
        steps_obj="alpha_max",
        init_bal=0.95,
        a_s_cut=0.6,
        lr=0.5,
        error_margin_alphamax=0.01,
        init_step_size=None,
        tune_stepsize=True,
    ):
        bdmala = LangevinSamplerOrdinal(
            dim=self.dim,
            n_steps=self.n_steps,
            multi_hop=self.multi_hop,
            step_size=1,
            mh=True,
            bal=init_bal,
        )
        if not init_step_size:
            init_step_size = (self.dim / 2) ** 0.5
        res = estimate_alpha_max(
            model,
            bdmala,
            x_init,
            budget,
            init_step_size,
            a_s_cut=a_s_cut,
            lr=lr,
            test_steps=test_steps,
            init_bal=init_bal,
            error_margin=error_margin_alphamax,
        )

        x_cur, final_step_size, hist_metrics, itr = res
        return x_cur, final_step_size, hist_metrics, itr

    # not going to edit this -- this is what provided the sampling results, so best to not touch
    def run_adaptive_burnin(
        self,
        x_init,
        model,
        budget,
        test_steps=10,
        steps_obj="alpha_min",
        bal_est_resolution=3,
        init_bal=0.95,
        a_s_cut=0.6,
        lr=0.5,
        error_margin_alphamax=0.01,
        error_margin_a_s_min=0.01,
        error_margin_hops_min=5,
        decrease_val_decay=0.5,
        init_step_size=None,
        tune_stepsize=True,
    ):
        bdmala = LangevinSamplerOrdinal(
            dim=self.dim,
            n_steps=self.n_steps,
            multi_hop=self.multi_hop,
            step_size=1,
            mh=True,
            bal=init_bal,
            max_val=self.max_val,
        )
        total_res = {}
        if not init_step_size:
            init_step_size = (2 * self.max_val**2) ** 0.5
        if tune_stepsize:
            if steps_obj == "alpha_max":
                res = estimate_alpha_max(
                    model,
                    bdmala,
                    x_init,
                    budget,
                    init_step_size,
                    a_s_cut=a_s_cut,
                    lr=lr,
                    test_steps=test_steps,
                    init_bal=init_bal,
                    error_margin=error_margin_alphamax,
                )
            else:
                res = estimate_alpha_min(
                    model,
                    bdmala,
                    x_init,
                    budget,
                    init_step_size,
                    test_steps=test_steps,
                    error_margin_a_s=error_margin_a_s_min,
                    error_margin_hops=error_margin_hops_min,
                    decrease_val_decay=decrease_val_decay,
                    init_bal=init_bal,
                )
            x_cur, final_step_size, hist_metrics, itr = res
            total_res["step-adapt-hist"] = hist_metrics
        else:
            final_step_size = init_step_size
            x_cur = x_init
        opt_steps = self.calc_stepsizes(final_step_size)
        x_cur, opt_bal, hist_metrics_bal = estimate_opt_bal(
            model,
            bdmala,
            x_cur,
            opt_steps,
            test_steps=test_steps,
            init_bal=init_bal,
            est_resolution=bal_est_resolution,
        )
        total_res["bal-adapt-hist"] = hist_metrics_bal
        self.step_sizes = opt_steps
        self.balancing_constants = opt_bal
        if self.min_lr:
            self.min_lr_cutoff()
        print(f"steps: {self.step_sizes}")
        print(f"bal: {self.balancing_constants}")
        return x_cur, total_res

    def get_grad(self, x, model):
        x = x.requires_grad_()
        out = model(x)
        gx = torch.autograd.grad(out.sum(), x)[0]
        return gx.detach()

    def _calc_logits(self, x_cur, grad, step_size, bal):
        # creating the tensor of discrete values to compute the probabilities for
        batch_size = x_cur.shape[0]
        disc_values = torch.tensor([i for i in range(self.max_val)])[None, None, :]
        disc_values = disc_values.repeat((batch_size, self.dim, 1)).to(x_cur.device)
        term1 = torch.zeros((batch_size, self.dim, self.max_val))
        term2 = torch.zeros((batch_size, self.dim, self.max_val))
        x_expanded = x_cur[:, :, None].repeat((1, 1, 64)).to(x_cur.device)
        grad_expanded = grad[:, :, None].repeat((1, 1, 64)).to(x_cur.device)
        term1 = grad_expanded * (disc_values - x_expanded) * bal
        term2 = (disc_values - x_expanded) ** 2 * (1 / (2 * step_size))
        return term1 - term2

    def step(self, x, model, k_iter):
        x_cur = x

        step_size = self.step_sizes[k_iter % self.iter_per_cycle]
        balancing_constant = self.balancing_constants[k_iter % self.iter_per_cycle]
        for i in range(self.n_steps):
            grad = self.get_grad(x_cur.float(), model)
            logits = self._calc_logits(
                x_cur, grad, step_size=step_size, bal=balancing_constant
            )
            cat_dist = torch.distributions.categorical.Categorical(logits=logits)
            x_delta = cat_dist.sample()
            if self.mh:
                lp_forward = torch.sum(cat_dist.log_prob(x_delta), dim=1)
                grad_delta = self.get_grad(x_delta.float(), model) / self.temp

                logits_delta = self._calc_logits(
                    x_delta, grad_delta, step_size=step_size, bal=balancing_constant
                )

                cat_dist_delta = torch.distributions.categorical.Categorical(
                    logits=logits_delta
                )
                lp_reverse = torch.sum(cat_dist_delta.log_prob(x_cur), dim=1)

                m_term = model(x_delta).squeeze() - model(x_cur).squeeze()
                la = m_term + lp_reverse - lp_forward
                a = (la.exp() > torch.rand_like(la)).float()
                self.a_s.append(a.mean().item())
                x_cur = x_delta * a[:, None] + x_cur * (1.0 - a[:, None])
            else:
                x_cur = x_delta
            x_cur = x_cur.long()
        return x_cur
