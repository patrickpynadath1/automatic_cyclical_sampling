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
        mean_stepsize_actual = torch.Tensor([mean_stepsize * self.max_val]).to(
            self.device
        ) ** (self.dim**2)
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
        actual_val = torch.Tensor([self.min_lr * self.max_val]).to(self.device) ** (
            self.dim**2
        )
        # actual_val = self.min_lr
        for i in range(len(self.step_sizes)):
            if self.step_sizes[i] <= actual_val:
                self.step_sizes[i] = actual_val
                self.balancing_constants[i] = 0.5

    def calc_stepsizes(self, mean_step):
        res = []
        total_iter = self.iter_per_cycle
        for k_iter in range(total_iter):
            inner = (np.pi * k_iter) / total_iter
            cur_step_size = mean_step * (np.cos(inner) + 1)
            step_size = cur_step_size
            res.append(step_size)
        res = (torch.tensor(res, device=self.device) * self.max_val) ** (self.dim**2)
        return res
        # return torch.Tensor(res).to(self.device)

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
    def adapt_alg_greedy_mod(
        self,
        dula_x_init,
        model,
        budget,
        init_big_step,
        init_small_step,
        init_big_bal=0.95,
        init_small_bal=0.5,
        a_s_cut=0.5,
        test_steps=10,
        lr=0.5,
        step_zoom_res=5,
        step_size_pair=None,
        x_init_to_use="bal",
        bal_resolution=3,
    ):
        bdmala = LangevinSamplerOrdinal(
            dim=self.dim,
            n_steps=self.n_steps,
            multi_hop=self.multi_hop,
            step_size=1,
            mh=True,
            bal=init_small_bal,
        )
        # pre burn in
        bdmala.mh = False
        bdmala.step_size = init_big_step
        bdmala.bal = init_big_bal
        bal_x_init = dula_x_init
        for i in range(100):
            dula_x_init = bdmala.step(dula_x_init, model).detach()
        bdmala.mh = True
        total_res = {}
        possible_x_inits = [dula_x_init]
        # estimating alpha min
        use_dula = not self.mh

        def step_update(sampler, alpha):
            sampler.step_size = alpha * (self.max_val**2) ** self.dim

        alpha_min_x_init, alpha_min, alpha_min_metrics, itr = estimate_alpha_min(
            model=model,
            bdmala=bdmala,
            x_cur=dula_x_init,
            budget=budget // 2,
            init_step_size=init_small_step,
            test_steps=test_steps,
            lr=lr,
            a_s_cut=a_s_cut,
            init_bal=init_small_bal,
            use_dula=use_dula,
            step_update=step_update,
        )
        print(alpha_min)
        possible_x_inits.append(alpha_min_x_init)
        (alpha_max_x_init, alpha_max, alpha_max_metrics, itrr) = estimate_alpha_max(
            model=model,
            bdmala=bdmala,
            a_s_cut=a_s_cut,
            init_bal=init_big_bal,
            test_steps=test_steps,
            budget=budget // 2,
            init_step_size=2,
            x_init=alpha_min_x_init,
            use_dula=use_dula,
            step_update=step_update,
        )

        init_big_bal = bdmala.bal
        print(init_big_bal)
        possible_x_inits.append(alpha_max_x_init)
        total_res["alpha_max_metrics"] = alpha_max_metrics
        total_res["alpha_min_metrics"] = alpha_min_metrics

        opt_steps = self.calc_stepsizes(alpha_max / 2)
        for i in range(len(opt_steps)):
            if opt_steps[i] < alpha_min:
                break
        bal_x_init, opt_bal, bal_metrics = estimate_opt_bal(
            model=model,
            bdmala=bdmala,
            x_init=dula_x_init,
            init_bal=init_big_bal,
            opt_steps=opt_steps[:i],
            est_resolution=bal_resolution,
            test_steps=test_steps,
            use_dula=use_dula,
        )
        possible_x_inits.append(bal_x_init)

        while i < len(opt_steps):
            opt_steps[i] = alpha_min
            opt_bal.append(init_small_bal)
            i += 1
        self.balancing_constants = opt_bal
        self.step_sizes = opt_steps
        total_res["bal_metrics"] = bal_metrics
        print("step sizes: \n")
        print(self.step_sizes)
        print("\n")
        print("bal: \n")
        print(self.balancing_constants)
        return dula_x_init, total_res

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
        x_expanded = x_cur[:, :, None].repeat((1, 1, self.max_val)).to(x_cur.device)
        grad_expanded = grad[:, :, None].repeat((1, 1, self.max_val)).to(x_cur.device)
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
