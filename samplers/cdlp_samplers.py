import math
import torch
import torch.nn as nn
import utils
import numpy as np
from .adaptive_components import *
from .dlp_samplers import LangevinSampler


class CyclicalLangevinSampler(nn.Module):
    def __init__(
        self,
        dim,
        device,
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
        min_lr=None,
        a_s_cut=None,
    ):
        super().__init__()
        self.device = device
        self.dim = dim
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
        if approx:
            self.diff_fn = lambda x, m: utils.approx_difference_function(x, m)
        else:
            self.diff_fn = lambda x, m: utils.difference_function(x, m)
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
        self.a_s_cut = a_s_cut

    def get_name(self):
        if self.mh:
            base = "cyc_dmala"
        else:
            base = "cyc_dula"
        if self.burnin_adaptive:
            name = f"{base}_cycles_{self.num_cycles}"
            name = (
                name
                + f"_budget_{self.burnin_budget}_lr_{self.burnin_lr}_a_s_cut_{self.a_s_cut}"
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
            name += f"_min_lr_{self.min_lr}"
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

    def calc_stepsizes_mod(self, mean_step):
        res = []
        total_iter = self.iter_per_cycle // 2
        for k_iter in range(total_iter):
            inner = (np.pi * k_iter) / total_iter
            cur_step_size = mean_step * (np.cos(inner) + 1)
            step_size = cur_step_size
            res.append(step_size)
        res = torch.tensor(res + [0 for _ in res], device=self.device)
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

    def adapt_bayes_gp(
        self,
        dula_x_init,
        model,
        budget,
        init_big_step,
        init_small_step,
        init_big_bal=0.95,
        init_small_bal=0.5,
        big_a_s_cut=None,
        small_a_s_cut=0.5,
        test_steps=30,
        lr=0.5,
        step_zoom_res=5,
        step_size_pair=None,
        step_schedule="mod",
        x_init_to_use="alpha_max",
        pair_optim=False,
        bal_resolution=3,
    ):
        bdmala = LangevinSampler(
            dim=self.dim,
            n_steps=self.n_steps,
            approx=self.approx,
            multi_hop=self.multi_hop,
            fixed_proposal=self.fixed_proposal,
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
        b_opt = BayesOptimizer(model, bdmala, test_steps)
        alpha_min, a_s_log, hops_log, min_x_init = b_opt.find_alpha_min(
            x_init=dula_x_init,
            target_a_s=small_a_s_cut,
            min_bal=init_small_bal,
            budget=budget // 2,
        )

        alpha_max, beta_max, a_s_log, hops_log, max_x_init = b_opt.find_max_pair(
            x_init=min_x_init, alpha_max=init_big_step, budget=budget // 2
        )
        alpha_max = alpha_max / 2
        print(alpha_max)
        print(beta_max)
        if step_schedule == "mod":
            opt_steps = self.calc_stepsizes_mod(alpha_max)
        else:
            opt_steps = self.calc_stepsizes(alpha_max)

        for i in range(len(opt_steps)):
            if opt_steps[i] < alpha_min:
                break
        bal_x_init, opt_bal, bal_metrics = estimate_opt_bal(
            model=model,
            bdmala=bdmala,
            x_init=max_x_init,
            init_bal=beta_max,
            opt_steps=opt_steps[:i],
            est_resolution=bal_resolution,
        )

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
        return bal_x_init, total_res

    def adapt_alg_greedy(
        self,
        dula_x_init,
        model,
        budget,
        init_big_step,
        init_small_step,
        init_big_bal=0.95,
        init_small_bal=0.5,
        big_a_s_cut=None,
        small_a_s_cut=0.5,
        test_steps=10,
        lr=0.5,
        step_zoom_res=5,
        step_size_pair=None,
        step_schedule="mod",
        x_init_to_use="alpha_max",
        pair_optim=False,
        bal_resolution=3,
    ):
        bdmala = LangevinSampler(
            dim=self.dim,
            n_steps=self.n_steps,
            approx=self.approx,
            multi_hop=self.multi_hop,
            fixed_proposal=self.fixed_proposal,
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
        if step_size_pair is None:
            # estimating alpha min
            alpha_min_x_init, alpha_min, alpha_min_metrics, itr = estimate_alpha_min(
                model=model,
                bdmala=bdmala,
                x_cur=dula_x_init,
                budget=budget // 2,
                init_step_size=init_small_step,
                test_steps=test_steps,
                lr=lr,
                a_s_cut=small_a_s_cut,
                init_bal=init_small_bal,
            )
            possible_x_inits.append(alpha_min_x_init)
            if pair_optim:
                # estimating alpha max
                (
                    alpha_max_x_init,
                    alpha_max,
                    alpha_max_metrics,
                    itr,
                ) = estimate_opt_pair_greedy(
                    model=model,
                    bdmala=bdmala,
                    x_init=dula_x_init,
                    range_max=init_big_step,
                    range_min=init_small_step,
                    a_s_cut=big_a_s_cut,
                    budget=budget // 2,
                    zoom_resolution=step_zoom_res,
                    test_steps=test_steps,
                    init_bal=init_big_bal,
                )
                init_big_bal = bdmala.bal
                print(init_big_bal)
            else:
                # estimating alpha max
                (
                    alpha_max_x_init,
                    alpha_max,
                    alpha_max_metrics,
                    itr,
                ) = estimate_opt_step_greedy(
                    model=model,
                    bdmala=bdmala,
                    x_init=dula_x_init,
                    range_max=init_big_step,
                    range_min=init_small_step,
                    a_s_cut=big_a_s_cut,
                    budget=budget // 2,
                    zoom_resolution=step_zoom_res,
                    test_steps=test_steps,
                    init_bal=init_big_bal,
                )
            possible_x_inits.append(alpha_max_x_init)
            total_res["alpha_max_metrics"] = alpha_max_metrics
            total_res["alpha_min_metrics"] = alpha_min_metrics
        else:
            alpha_max = step_size_pair[0]
            alpha_min = step_size_pair[1]

        if step_schedule == "mod":
            opt_steps = self.calc_stepsizes_mod(alpha_max)
        else:
            opt_steps = self.calc_stepsizes(alpha_max)

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
        if x_init_to_use == "alpha_min" and step_size_pair is not None:
            x_init = possible_x_inits[1]
        elif x_init_to_use == "alpha_max" and step_size_pair is not None:
            x_init = possible_x_inits[2]
        elif x_init_to_use == "bal":
            x_init = possible_x_inits[3]
        else:
            x_init = possible_x_inits[0]
        return x_init, total_res

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
        bdmala = LangevinSampler(
            dim=self.dim,
            n_steps=self.n_steps,
            approx=self.approx,
            multi_hop=self.multi_hop,
            fixed_proposal=self.fixed_proposal,
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
        bal_est_resolution=6,
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
        bdmala = LangevinSampler(
            dim=self.dim,
            n_steps=self.n_steps,
            approx=self.approx,
            multi_hop=self.multi_hop,
            fixed_proposal=self.fixed_proposal,
            step_size=1,
            mh=True,
            bal=init_bal,
        )
        total_res = {}
        if not init_step_size:
            init_step_size = (self.dim / 2) ** 0.5
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

    def step(self, x, model, k_iter, return_diff=False):
        x_cur = x

        m_terms = []
        prop_terms = []

        EPS = 1e-10
        step_size = self.step_sizes[k_iter % self.iter_per_cycle]
        balancing_constant = self.balancing_constants[k_iter % self.iter_per_cycle]

        for i in range(self.n_steps):
            forward_delta = self.diff_fn(x_cur, model)
            forward_delta_bal = forward_delta * balancing_constant
            term2 = 1.0 / (2 * step_size)  # for binary {0,1}, the L2 norm is always 1
            flip_prob = torch.exp(forward_delta_bal - term2) / (
                torch.exp(forward_delta_bal - term2) + 1
            )
            self.flip_probs.append(flip_prob.detach().sum(axis=-1).mean().item())
            rr = torch.rand_like(x_cur)
            ind = (rr < flip_prob) * 1
            x_delta = (1.0 - x_cur) * ind + x_cur * (1.0 - ind)
            if self.mh:
                probs = flip_prob * ind + (1 - flip_prob) * (1.0 - ind)
                lp_forward = torch.sum(torch.log(probs + EPS), dim=-1)

                reverse_delta = self.diff_fn(x_delta, model) * balancing_constant
                flip_prob = torch.exp(reverse_delta - term2) / (
                    torch.exp(reverse_delta - term2) + 1
                )
                probs = flip_prob * ind + (1 - flip_prob) * (1.0 - ind)
                lp_reverse = torch.sum(torch.log(probs + EPS), dim=-1)
                m_term = model(x_delta).squeeze() - model(x_cur).squeeze()
                la = m_term + lp_reverse - lp_forward
                a = (la.exp() > torch.rand_like(la)).float()
                x_cur = x_delta * a[:, None] + x_cur * (1.0 - a[:, None])
                probs_tmp = torch.minimum(
                    torch.ones_like(la, device=la.device), la.exp()
                )
                self.a_s.append(probs_tmp.detach().mean().item())
            else:
                x_cur = x_delta
        if return_diff:
            probs = torch.minimum(torch.ones_like(la, device=la.device), la.exp())
            return x_cur, forward_delta, probs
        else:
            return x_cur
