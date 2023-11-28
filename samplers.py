import math

import torch
import torch.nn as nn
import torch.distributions as dists
import utils
import numpy as np

device = torch.device("cuda:" + str(0) if torch.cuda.is_available() else "cpu")


class LangevinSampler(nn.Module):
    def __init__(
        self,
        dim,
        n_steps=10,
        approx=False,
        multi_hop=False,
        fixed_proposal=False,
        temp=1.0,
        step_size=0.2,
        mh=True,
        bal=0.5,
        use_big=False,
        burn_in_adaptive=False,
        burn_in_budget=500,
        adapt_alg="twostage_optim",
    ):
        super().__init__()
        self.dim = dim
        self.n_steps = n_steps
        self._ar = 0.0
        self._mt = 0.0
        self._pt = 0.0
        self._hops = 0.0
        self.adapt_alg = adapt_alg
        self.burn_in_budget = burn_in_budget
        self.burn_in_adaptive = burn_in_adaptive
        self._phops = 0.0
        self.approx = approx
        self.fixed_proposal = fixed_proposal
        self.multi_hop = multi_hop
        self.temp = temp
        self.step_size = step_size
        self.bal = bal

        if approx:
            self.diff_fn = (
                lambda x, m: self.bal
                * utils.approx_difference_function(x, m)
                / self.temp
            )
        else:
            self.diff_fn = (
                lambda x, m: self.bal * utils.difference_function(x, m) / self.temp
            )

        self.mh = mh
        self.a_s = []
        self.hops = []
        self.use_big = use_big

    def get_name(self):
        if self.mh:
            base = "dmala"

        else:
            base = "dula"
        if self.use_big:
            name = f"{base}_use_big"
        elif self.burn_in_adaptive:
            name = f"{base}_{self.adapt_alg}_budget_{self.burn_in_budget}"
        else:
            name = f"{base}_stepsize_{self.step_size}_{self.bal}"

        return name

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
        num_bal_est=5,
        decrease_val_decay=0.5,
        init_step_size=None,
        mode="full",
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
        # if no initial step size passed in, we just reset it
        if not init_step_size:
            init_step_size = (self.dim / 2) ** 0.5
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
        self.step_size = final_step_size

        self.bal = init_bal

        return x_cur, {"step-adapt-hist": hist_metrics}

    def step(self, x, model):
        x_cur = x

        m_terms = []
        prop_terms = []

        EPS = 1e-10
        for i in range(self.n_steps):
            forward_delta = self.diff_fn(x_cur, model)
            if self.use_big:
                term2 = 0
            else:
                term2 = 1.0 / (
                    2 * self.step_size
                )  # for binary {0,1}, the L2 norm is always 1
            flip_prob = torch.exp(forward_delta - term2) / (
                torch.exp(forward_delta - term2) + 1
            )
            rr = torch.rand_like(x_cur)
            ind = (rr < flip_prob) * 1
            x_delta = (1.0 - x_cur) * ind + x_cur * (1.0 - ind)

            if self.mh:
                probs = flip_prob * ind + (1 - flip_prob) * (1.0 - ind)
                lp_forward = torch.sum(torch.log(probs + EPS), dim=-1)

                reverse_delta = self.diff_fn(x_delta, model)
                flip_prob = torch.exp(reverse_delta - term2) / (
                    torch.exp(reverse_delta - term2) + 1
                )
                probs = flip_prob * ind + (1 - flip_prob) * (1.0 - ind)
                lp_reverse = torch.sum(torch.log(probs + EPS), dim=-1)

                m_term = model(x_delta).squeeze() - model(x_cur).squeeze()
                la = m_term + lp_reverse - lp_forward
                a = (la.exp() > torch.rand_like(la)).float()
                # self.a_s.append(a.mean().item())
                probs_tmp = torch.minimum(
                    torch.ones_like(la, device=la.device), la.exp()
                )
                self.a_s.append(probs_tmp.detach().mean().item())
                x_cur = x_delta * a[:, None] + x_cur * (1.0 - a[:, None])
            else:
                x_cur = x_delta
        return x_cur


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
        min_lr=False,
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

        # need two arrays: one for explore step sizes, another for exploit step sizes
        self.min_lr = min_lr
        if self.min_lr:
            self.min_lr_cutoff()
        else:
            self.sbc = False

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
        if min_lr:
            name += "_min_lr"
        return name

    def min_lr_cutoff(self):
        for i, step in self.stepsizes:
            if step <= 0.1:
                self.step_sizes[i] = 0.1
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

    # this is the older version of the adaptive function -- keeping for posterity
    def run_burn_in_bb_modified(
        self,
        x_init,
        model,
        acc_cut=0.6,
        test_steps=10,
        lr=0.5,
        init_bal=0.95,
        use_rev_bal=True,
        use_hops=True,
    ):
        print(f"use rev bal: {use_rev_bal}")
        print(f"use hops: {use_hops}")
        cur_step = (self.dim / 2) ** 0.5
        cur_acc = 0

        # adding a burn in for the burn in -- seems redundant, but think it is necessary
        x_cur = x_init
        bdmala = LangevinSampler(
            dim=self.dim,
            n_steps=self.n_steps,
            approx=self.approx,
            multi_hop=self.multi_hop,
            fixed_proposal=self.fixed_proposal,
            step_size=cur_step,
            mh=True,
            bal=init_bal,
        )
        itr = 0

        hist_steps_alpha_max = []
        while itr < 50:
            proposal_step = cur_step * (1 - lr * np.abs(cur_acc - acc_cut))
            steps_to_test = [proposal_step, cur_step]
            x_init = x_cur
            accs = []
            pot_pos = []
            for s in steps_to_test:
                bdmala.step_size = s
                bdmala.a_s = []
                x_cur = x_init
                for _ in range(test_steps):
                    x_cur = bdmala.step(x_cur.detach(), model)
                cur_acc = np.mean(bdmala.a_s)
                accs.append(cur_acc)
                pot_pos.append(x_cur)

            best = np.argmax(accs)
            cur_step = steps_to_test[best]
            x_cur = pot_pos[best]
            cur_acc = accs[best]
            hist_steps_alpha_max.append(cur_step)
            if np.abs(cur_acc - acc_cut) < 0.01 and cur_acc > acc_cut:
                break

            itr += 1
        hist_steps_alpha_min = []
        if use_hops:
            # decreasing steps until it starts to affect the hops
            itr = 0
            decrease_val = 1
            while itr < 100:
                proposal_step = cur_step - decrease_val
                steps_to_test = [proposal_step, cur_step]
                x_init = x_cur
                accs = []
                hops = []
                for s in steps_to_test:
                    bdmala.step_size = s
                    bdmala.a_s = []
                    x_cur = x_init
                    h = []
                    for _ in range(test_steps):
                        x_new = bdmala.step(x_cur.detach(), model)
                        cur_hops = (x_new != x_cur).float().sum(-1).mean().item()
                        h.append(cur_hops)
                        itr += 1
                    accs.append(np.mean(bdmala.a_s))
                    hops.append(np.mean(cur_hops))

                if np.abs(hops[0] - hops[1]) < 5 and np.abs(accs[0] - accs[1]) < 0.02:
                    cur_step = proposal_step
                    hist_steps_alpha_min.append(cur_step)
                else:
                    decrease_val /= 2

        opt_steps = self.calc_stepsizes(cur_step / 1)
        opt_bal = []
        if use_rev_bal:
            bal_proposals = [0.5, 0.55, 0.6]
        else:
            bal_proposals = [init_bal, init_bal - 0.05, init_bal - 0.1]
        best_bal_accs = []
        for i in range(len(opt_steps)):
            accs = []
            x_init = x_cur
            for b in bal_proposals:
                x_cur = x_init
                bdmala.a_s = []
                if use_rev_bal:
                    bdmala.step_size = opt_steps[-i - 1]
                else:
                    bdmala.step_size = opt_steps[i]
                bdmala.bal = b
                for _ in range(test_steps):
                    x_cur = bdmala.step(x_cur.detach(), model)
                accs.append(np.mean(bdmala.a_s))
            best_idx = np.argmax(accs)
            best_bal_accs.append(accs[best_idx])
            best_bal = bal_proposals[best_idx]
            opt_bal.append(best_bal)
            if use_rev_bal:
                bal_proposals = np.linspace(best_bal, min(init_bal, best_bal + 0.25), 6)
            else:
                bal_proposals = np.linspace(best_bal, max(0.5, best_bal - 0.25), 6)

        self.step_sizes = opt_steps
        if use_rev_bal:
            self.balancing_constants = opt_bal[::-1]
        else:
            self.balancing_constants = opt_bal
        return [hist_steps_alpha_max, hist_steps_alpha_min], self.balancing_constants

    # this is the first adaptive burn in algorithm that worked decently well -- keeping it for posterity
    # DO NOT TOUCH OR EDIT THIS FUNCTION UNTIL THE OTHER ONE WORKS PERFECTLY
    def run_burn_in_bb(
        self, x_init, model, acc_cut=0.6, test_steps=10, lr=0.1, init_bal=0.95
    ):
        cur_step = (self.dim / 2) ** 0.5
        cur_acc = -np.infty

        # adding a burn in for the burn in -- seems redundant, but think it is necessary
        x_cur = x_init
        bdmala = LangevinSampler(
            dim=self.dim,
            n_steps=self.n_steps,
            approx=self.approx,
            multi_hop=self.multi_hop,
            fixed_proposal=self.fixed_proposal,
            step_size=cur_step,
            mh=True,
            bal=init_bal,
        )
        itr = 0
        hist_steps = []
        while itr < 50:
            if itr == 0:
                proposal_step = cur_step / 2
            else:
                proposal_step = cur_step * (1 - lr * np.abs(cur_acc - acc_cut))
            steps_to_test = [proposal_step, cur_step]
            x_init = x_cur
            accs = []
            for s in steps_to_test:
                bdmala.step_size = s
                bdmala.a_s = []
                x_cur = x_init
                for _ in range(test_steps):
                    x_cur = bdmala.step(x_cur.detach(), model)
                cur_acc = np.mean(bdmala.a_s)
                accs.append(cur_acc)

            best = np.argmax(accs)
            cur_step = steps_to_test[best]
            cur_acc = accs[best]
            hist_steps.append(cur_step)
            if np.abs(cur_acc - acc_cut) < 0.01 and cur_acc > acc_cut:
                break

            itr += 1

        # decreasing steps until it starts to affect the hops
        itr = 0
        decrease_val = 1
        while itr < 100:
            proposal_step = cur_step - decrease_val
            steps_to_test = [proposal_step, cur_step]
            x_init = x_cur
            accs = []
            hops = []
            for s in steps_to_test:
                bdmala.step_size = s
                bdmala.a_s = []
                x_cur = x_init
                h = []
                for _ in range(test_steps):
                    x_new = bdmala.step(x_cur.detach(), model)
                    cur_hops = (x_new != x_cur).float().sum(-1).mean().item()
                    h.append(cur_hops)
                    itr += 1
                accs.append(np.mean(bdmala.a_s))
                hops.append(np.mean(cur_hops))

            if np.abs(hops[0] - hops[1]) < 5 and np.abs(accs[0] - accs[1]) < 0.02:
                cur_step = proposal_step
                hist_steps.append(cur_step)
            else:
                decrease_val /= 2
        opt_steps = self.calc_stepsizes(cur_step / 1)
        opt_bal = []

        bal_proposals = [0.5, 0.55, 0.6]
        best_bal_accs = []
        for i in range(len(opt_steps)):
            accs = []
            x_init = x_cur
            for b in bal_proposals:
                x_cur = x_init
                bdmala.a_s = []
                bdmala.step_size = opt_steps[-i - 1]
                bdmala.bal = b
                for _ in range(test_steps):
                    x_cur = bdmala.step(x_cur.detach(), model)
                accs.append(np.mean(bdmala.a_s))
            best_idx = np.argmax(accs)
            best_bal_accs.append(accs[best_idx])
            best_bal = bal_proposals[best_idx]
            opt_bal.append(best_bal)
            bal_proposals = np.linspace(best_bal, min(init_bal, best_bal + 0.25), 6)
        self.step_sizes = opt_steps
        self.balancing_constants = opt_bal[::-1]
        return hist_steps, self.balancing_constants

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


#


# Gibbs-With-Gradients for binary data
class DiffSampler(nn.Module):
    def __init__(
        self,
        dim,
        n_steps=10,
        approx=False,
        multi_hop=False,
        fixed_proposal=False,
        temp=2.0,
        step_size=1.0,
    ):
        super().__init__()
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
        self.step_size = step_size
        if approx:
            self.diff_fn = (
                lambda x, m: utils.approx_difference_function(x, m) / self.temp
            )
        else:
            self.diff_fn = lambda x, m: utils.difference_function(x, m) / self.temp

    def step(self, x, model):
        x_cur = x
        a_s = []
        m_terms = []
        prop_terms = []

        if self.multi_hop:
            if self.fixed_proposal:
                delta = self.diff_fn(x, model)
                cd = dists.Bernoulli(probs=delta.sigmoid() * self.step_size)
                for i in range(self.n_steps):
                    changes = cd.sample()
                    x_delta = (1.0 - x_cur) * changes + x_cur * (1.0 - changes)
                    la = model(x_delta).squeeze() - model(x_cur).squeeze()
                    a = (la.exp() > torch.rand_like(la)).float()
                    x_cur = x_delta * a[:, None] + x_cur * (1.0 - a[:, None])
                    a_s.append(a.mean().item())
                self._ar = np.mean(a_s)
            else:
                for i in range(self.n_steps):
                    forward_delta = self.diff_fn(x_cur, model)
                    cd_forward = dists.Bernoulli(logits=(forward_delta * 2 / self.temp))
                    changes = cd_forward.sample()

                    lp_forward = cd_forward.log_prob(changes).sum(-1)

                    x_delta = (1.0 - x_cur) * changes + x_cur * (1.0 - changes)

                    reverse_delta = self.diff_fn(x_delta, model)
                    cd_reverse = dists.Bernoulli(logits=(reverse_delta * 2 / self.temp))

                    lp_reverse = cd_reverse.log_prob(changes).sum(-1)

                    m_term = model(x_delta).squeeze() - model(x_cur).squeeze()
                    la = m_term + lp_reverse - lp_forward
                    a = (la.exp() > torch.rand_like(la)).float()
                    x_cur = x_delta * a[:, None] + x_cur * (1.0 - a[:, None])
                    a_s.append(a.mean().item())
                    m_terms.append(m_term.mean().item())
                    prop_terms.append((lp_reverse - lp_forward).mean().item())
                self._ar = np.mean(a_s)
                self._mt = np.mean(m_terms)
                self._pt = np.mean(prop_terms)
        else:
            if self.fixed_proposal:
                delta = self.diff_fn(x, model)
                cd = dists.OneHotCategorical(logits=delta)
                for i in range(self.n_steps):
                    changes = cd.sample()

                    x_delta = (1.0 - x_cur) * changes + x_cur * (1.0 - changes)
                    la = model(x_delta).squeeze() - model(x_cur).squeeze()
                    a = (la.exp() > torch.rand_like(la)).float()
                    x_cur = x_delta * a[:, None] + x_cur * (1.0 - a[:, None])
                    a_s.append(a.mean().item())
                self._ar = np.mean(a_s)
            else:
                for i in range(self.n_steps):
                    forward_delta = self.diff_fn(x_cur, model)
                    cd_forward = dists.OneHotCategorical(logits=forward_delta)
                    changes = cd_forward.sample()

                    lp_forward = cd_forward.log_prob(changes)

                    x_delta = (1.0 - x_cur) * changes + x_cur * (1.0 - changes)

                    reverse_delta = self.diff_fn(x_delta, model)
                    cd_reverse = dists.OneHotCategorical(logits=reverse_delta)

                    lp_reverse = cd_reverse.log_prob(changes)

                    m_term = model(x_delta).squeeze() - model(x_cur).squeeze()
                    la = m_term + lp_reverse - lp_forward
                    a = (la.exp() > torch.rand_like(la)).float()
                    x_cur = x_delta * a[:, None] + x_cur * (1.0 - a[:, None])

        return x_cur


# Gibbs-With-Gradients variant which proposes multiple flips per step
class MultiDiffSampler(nn.Module):
    def __init__(self, dim, n_steps=10, approx=False, temp=1.0, n_samples=1):
        super().__init__()
        self.dim = dim
        self.n_steps = n_steps
        self._ar = 0.0
        self._mt = 0.0
        self._pt = 0.0
        self._hops = 0.0
        self._phops = 0.0
        self.approx = approx
        self.temp = temp
        self.n_samples = n_samples
        if approx:
            self.diff_fn = (
                lambda x, m: utils.approx_difference_function(x, m) / self.temp
            )
        else:
            self.diff_fn = lambda x, m: utils.difference_function(x, m) / self.temp
        self.a_s = []
        self.hops = []

    def step(self, x, model):
        x_cur = x
        a_s = []
        m_terms = []
        prop_terms = []

        for i in range(self.n_steps):
            forward_delta = self.diff_fn(x_cur, model)
            cd_forward = dists.OneHotCategorical(logits=forward_delta)
            changes_all = cd_forward.sample((self.n_samples,))

            lp_forward = cd_forward.log_prob(changes_all).sum(0)

            changes = (changes_all.sum(0) > 0.0).float()

            x_delta = (1.0 - x_cur) * changes + x_cur * (1.0 - changes)
            # self._phops = (x_delta != x).float().sum(-1).mean().item()
            cur_hops = (x_cur[0] != x_delta[0]).float().sum(-1).item()
            self.hops.append(cur_hops)

            reverse_delta = self.diff_fn(x_delta, model)
            cd_reverse = dists.OneHotCategorical(logits=reverse_delta)

            lp_reverse = cd_reverse.log_prob(changes_all).sum(0)

            m_term = model(x_delta).squeeze() - model(x_cur).squeeze()
            la = m_term + lp_reverse - lp_forward
            a = (la.exp() > torch.rand_like(la)).float()
            x_cur = x_delta * a[:, None] + x_cur * (1.0 - a[:, None])
            self.a_s.append(a.mean().item())
            m_terms.append(m_term.mean().item())
            prop_terms.append((lp_reverse - lp_forward).mean().item())
        self._ar = np.mean(a_s)
        self._mt = np.mean(m_terms)
        self._pt = np.mean(prop_terms)
        # print(self._ar)
        self._hops = (x != x_cur).float().sum(-1).mean().item()
        return x_cur


class PerDimGibbsSampler(nn.Module):
    def __init__(self, dim, rand=False):
        super().__init__()
        self.dim = dim
        self.changes = torch.zeros((dim,))
        self.change_rate = 0.0
        self.p = nn.Parameter(torch.zeros((dim,)))
        self._i = 0
        self._ar = 0.0
        self._hops = 0.0
        self._phops = 1.0
        self.rand = rand

    def step(self, x, model):
        sample = x.clone()
        lp_keep = model(sample).squeeze()
        if self.rand:
            changes = (
                dists.OneHotCategorical(logits=torch.zeros((self.dim,)))
                .sample((x.size(0),))
                .to(x.device)
            )
        else:
            changes = torch.zeros((x.size(0), self.dim)).to(x.device)
            changes[:, self._i] = 1.0

        sample_change = (1.0 - changes) * sample + changes * (1.0 - sample)

        lp_change = model(sample_change).squeeze()

        lp_update = lp_change - lp_keep
        update_dist = dists.Bernoulli(logits=lp_update)
        updates = update_dist.sample()
        sample = sample_change * updates[:, None] + sample * (1.0 - updates[:, None])
        self.changes[self._i] = updates.mean()
        self._i = (self._i + 1) % self.dim
        self._hops = (x != sample).float().sum(-1).mean().item()
        self._ar = self._hops
        return sample

    def logp_accept(self, xhat, x, model):
        # only true if xhat was generated from self.step(x, model)
        return 0.0


class PerDimMetropolisSampler(nn.Module):
    def __init__(self, dim, n_out, rand=False):
        super().__init__()
        self.dim = dim
        self.n_out = n_out
        self.changes = torch.zeros((dim,))
        self.change_rate = 0.0
        self.p = nn.Parameter(torch.zeros((dim,)))
        self._i = 0
        self._j = 0
        self._ar = 0.0
        self._hops = 0.0
        self._phops = 0.0
        self.rand = rand

    def step(self, x, model):
        if self.rand:
            i = np.random.randint(0, self.dim)
        else:
            i = self._i

        logits = []
        ndim = x.size(-1)

        for k in range(ndim):
            sample = x.clone()
            sample_i = torch.zeros((ndim,))
            sample_i[k] = 1.0
            sample[:, i, :] = sample_i
            lp_k = model(sample).squeeze()
            logits.append(lp_k[:, None])
        logits = torch.cat(logits, 1)
        dist = dists.OneHotCategorical(logits=logits)
        updates = dist.sample()
        sample = x.clone()
        sample[:, i, :] = updates
        self._i = (self._i + 1) % self.dim
        self._hops = ((x != sample).float().sum(-1) / 2.0).sum(-1).mean().item()
        self._ar = self._hops
        return sample

    def logp_accept(self, xhat, x, model):
        # only true if xhat was generated from self.step(x, model)
        return 0.0


class PerDimLB(nn.Module):
    def __init__(self, dim, rand=False):
        super().__init__()
        self.dim = dim
        self.changes = torch.zeros((dim,))
        self.change_rate = 0.0
        self.p = nn.Parameter(torch.zeros((dim,)))
        self._i = 0
        self._j = 0
        self._ar = 0.0
        self._hops = 0.0
        self._phops = 0.0
        self.rand = rand

    def step(self, x, model):
        logits = []
        ndim = x.size(-1)
        fx = model(x).squeeze()
        for k in range(ndim):
            sample = x.clone()
            sample[:, k] = 1 - sample[:, k]
            lp_k = (model(sample).squeeze() - fx) / 2.0
            logits.append(lp_k[:, None])
        logits = torch.cat(logits, 1)
        Z_forward = torch.sum(torch.exp(logits), dim=-1)
        dist = dists.OneHotCategorical(logits=logits)
        changes = dist.sample()
        x_delta = (1.0 - x) * changes + x * (1.0 - changes)
        fx_delta = model(x_delta)
        logits = []
        for k in range(ndim):
            sample = x_delta.clone()
            sample[:, k] = 1 - sample[:, k]
            lp_k = (model(sample).squeeze() - fx_delta) / 2.0
            logits.append(lp_k[:, None])
        logits = torch.cat(logits, 1)
        Z_reverse = torch.sum(torch.exp(logits), dim=-1)
        la = Z_forward / Z_reverse
        a = (la > torch.rand_like(la)).float()
        x = x_delta * a[:, None] + x * (1.0 - a[:, None])
        # a_s.append(a.mean().item())
        # self._ar = np.mean(a_s)
        return x

    def logp_accept(self, xhat, x, model):
        # only true if xhat was generated from self.step(x, model)
        return 0.0


# Gibbs-With-Gradients for categorical data
class DiffSamplerMultiDim(nn.Module):
    def __init__(self, dim, n_steps=10, approx=False, temp=1.0):
        super().__init__()
        self.dim = dim
        self.n_steps = n_steps
        self._ar = 0.0
        self._mt = 0.0
        self._pt = 0.0
        self._hops = 0.0
        self._phops = 0.0
        self.approx = approx
        self.temp = temp
        if approx:
            self.diff_fn = (
                lambda x, m: utils.approx_difference_function_multi_dim(x, m)
                / self.temp
            )
        else:
            self.diff_fn = (
                lambda x, m: utils.difference_function_multi_dim(x, m) / self.temp
            )

    def step(self, x, model):
        x_cur = x
        a_s = []
        m_terms = []
        prop_terms = []

        for i in range(self.n_steps):
            constant = 1.0
            forward_delta = self.diff_fn(x_cur, model)

            # make sure we dont choose to stay where we are!
            forward_logits = forward_delta - constant * x_cur
            # print(forward_logits)
            cd_forward = dists.OneHotCategorical(
                logits=forward_logits.view(x_cur.size(0), -1)
            )
            changes = cd_forward.sample()
            # print(x_cur.shape,forward_delta.shape,changes.shape)
            # exit()
            # compute probability of sampling this change
            lp_forward = cd_forward.log_prob(changes)
            # reshape to (bs, dim, nout)
            changes_r = changes.view(x_cur.size())
            # get binary indicator (bs, dim) indicating which dim was changed
            changed_ind = changes_r.sum(-1)
            # mask out cuanged dim and add in the change
            x_delta = x_cur.clone() * (1.0 - changed_ind[:, :, None]) + changes_r

            reverse_delta = self.diff_fn(x_delta, model)
            reverse_logits = reverse_delta - constant * x_delta
            cd_reverse = dists.OneHotCategorical(
                logits=reverse_logits.view(x_delta.size(0), -1)
            )
            reverse_changes = x_cur * changed_ind[:, :, None]

            lp_reverse = cd_reverse.log_prob(reverse_changes.view(x_delta.size(0), -1))

            m_term = model(x_delta).squeeze() - model(x_cur).squeeze()
            la = m_term + lp_reverse - lp_forward
            a = (la.exp() > torch.rand_like(la)).float()
            x_cur = x_delta * a[:, None, None] + x_cur * (1.0 - a[:, None, None])
            a_s.append(a.mean().item())
            m_terms.append(m_term.mean().item())
            prop_terms.append((lp_reverse - lp_forward).mean().item())
        self._ar = np.mean(a_s)
        self._mt = np.mean(m_terms)
        self._pt = np.mean(prop_terms)

        self._hops = (x != x_cur).float().sum(-1).sum(-1).mean().item()
        return x_cur


class GibbsSampler(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.changes = torch.zeros((dim,))
        self.change_rate = 0.0
        self.p = nn.Parameter(torch.zeros((dim,)))

    def step(self, x, model):
        sample = x.clone()
        for i in range(self.dim):
            lp_keep = model(sample).squeeze()

            xi_keep = sample[:, i]
            xi_change = 1.0 - xi_keep
            sample_change = sample.clone()
            sample_change[:, i] = xi_change

            lp_change = model(sample_change).squeeze()

            lp_update = lp_change - lp_keep
            update_dist = dists.Bernoulli(logits=lp_update)
            updates = update_dist.sample()
            sample = sample_change * updates[:, None] + sample * (
                1.0 - updates[:, None]
            )
            self.changes[i] = updates.mean()
        return sample

    def logp_accept(self, xhat, x, model):
        # only true if xhat was generated from self.step(x, model)
        return 0.0


def eval_bdmala(model, bdmala, x_init, test_steps):
    hops = []
    x_cur = x_init
    bdmala.a_s = []
    for _ in range(test_steps):
        x_new = bdmala.step(x_cur.detach(), model)
        cur_hops = (x_new != x_cur).float().sum(-1).mean().item()
        hops.append(cur_hops)
        x_cur = x_new
    a_s = np.mean(bdmala.a_s)
    hops_mean = np.mean(hops)
    return a_s, hops_mean, x_cur


def update_hyperparam_metrics(best_idx, *lists):
    to_ret = []
    for l in lists:
        to_ret.append(l[best_idx])
    return to_ret


# helper function for adaptive functions
# essentially, tests all the hyper-parameters of interest
def run_hyperparameters(
    model, bdmala, x_init, test_steps, param_update_func, param_list
):
    a_s = []
    hops = []
    x_potential = []
    for p in param_list:
        param_update_func(bdmala, p)
        cur_a_s, cur_hops, x_cur = eval_bdmala(model, bdmala, x_init, test_steps)
        a_s.append(cur_a_s)
        hops.append(cur_hops)
        x_potential.append(x_cur)
    return a_s, hops, x_potential


def estimate_alpha_max(
    model,
    bdmala,
    x_init,
    budget,
    init_step_size,
    a_s_cut=0.6,
    lr=0.5,
    test_steps=10,
    init_bal=0.95,
    error_margin=0.01,
):
    a_s = 0
    cur_step = init_step_size
    hist_a_s = []
    hist_alpha_max = []
    hist_hops = []

    x_cur = x_init
    itr = 0
    bdmala.bal = init_bal

    def step_update(sampler, alpha):
        sampler.step_size = alpha

    while itr < budget:
        proposal_step = cur_step * (1 - lr * np.abs(a_s - a_s_cut))
        steps_to_test = [proposal_step, cur_step]
        a_s_l, hops_l, x_potential_l = run_hyperparameters(
            model, bdmala, x_cur, test_steps, step_update, steps_to_test
        )
        cur_step, a_s, hops, x_cur = update_hyperparam_metrics(
            np.argmax(a_s_l), steps_to_test, a_s_l, hops_l, x_potential_l
        )
        itr += test_steps * len(steps_to_test)
        hist_a_s.append(a_s)
        hist_alpha_max.append(cur_step)
        hist_hops.append(hops)
        # modified to see if this improves performance and avoids the case where it just continually decreases and gets smaller, worsening acceptance rate
        if a_s_cut - a_s < error_margin:
            break
    final_step_size = cur_step / 2
    hist_metrics = {"a_s": hist_a_s, "hops": hist_hops, "alpha_max": hist_alpha_max}
    return x_cur, final_step_size, hist_metrics, itr


def estimate_alpha_min(
    model,
    bdmala,
    x_cur,
    budget,
    init_step_size,
    test_steps=10,
    error_margin_a_s=0.01,
    error_margin_hops=5,
    decrease_val_decay=0.5,
    init_bal=0.95,
):
    cur_step = init_step_size
    decrease_val = cur_step / 4
    # book keeping lists
    hist_a_s = []
    hist_alpha_min = []
    hist_hops = []
    # initialization for best acceptance rate, hops
    itr = 0

    def step_update(sampler, alpha):
        sampler.step_size = alpha

    while itr < budget:
        proposal_step = cur_step - decrease_val
        steps_to_test = [proposal_step, cur_step]

        a_s_l, hops_l, x_potential_l = run_hyperparameters(
            model, bdmala, x_cur, test_steps, step_update, steps_to_test
        )

        # we want the smallest_step size possible
        cur_step_min, a_s_min, hops_min, x_cur_min = update_hyperparam_metrics(
            np.argmin(steps_to_test), steps_to_test, a_s_l, hops_l, x_potential_l
        )

        cur_step_max, a_s_max, hops_max, x_cur_max = update_hyperparam_metrics(
            np.argmax(steps_to_test), steps_to_test, a_s_l, hops_l, x_potential_l
        )
        itr += len(steps_to_test) * test_steps
        # if selecting the smallest step leads to a strictly worse sampler, do not update
        if (
            a_s_max - a_s_min > error_margin_a_s
            or np.abs(hops_min - hops_max) > error_margin_hops
        ):
            decrease_val *= decrease_val_decay
            hist_a_s.append(a_s_max)
            hist_hops.append(hops_max)
            hist_alpha_min.append(cur_step_max)
            cur_step = cur_step_max
            x_cur = x_cur_max
        else:
            cur_step = cur_step_min
            hist_a_s.append(a_s_min)
            hist_hops.append(hops_min)
            x_cur = x_cur_min
            hist_alpha_min.append(cur_step)
            cur_step = cur_step_min
    final_step_size = cur_step / 2
    hist_metrics = {"a_s": hist_a_s, "hops": hist_hops, "alpha_min": hist_alpha_min}
    return x_cur, final_step_size, hist_metrics, itr


def estimate_opt_bal(
    model, bdmala, x_init, opt_steps, test_steps=10, init_bal=0.95, est_resolution=6
):
    # we will calculate in REVERSE order
    if type(opt_steps) == list:
        alpha_to_use = opt_steps[::-1]
    elif type(opt_steps) == torch.Tensor:
        alpha_to_use = opt_steps.flip(dims=(0,))
    opt_bal = []
    hist_a_s = []
    hist_hops = []
    x_cur = x_init
    bal_proposals = [0.5, 0.55, 0.6]

    def update_bal(sampler, beta):
        sampler.bal = beta

    for i, alpha in enumerate(alpha_to_use):
        bdmala.step_size = alpha
        a_s_l, hops_l, x_potential_l = run_hyperparameters(
            model, bdmala, x_cur, test_steps, update_bal, bal_proposals
        )
        best_bal, a_s, hops, x_cur = update_hyperparam_metrics(
            np.argmax(a_s_l), bal_proposals, a_s_l, hops_l, x_potential_l
        )
        hist_a_s.append(a_s)
        hist_hops.append(hops)
        opt_bal.append(best_bal)
        if best_bal >= 0.9:
            num_to_interpolate = len(opt_steps) - len(opt_bal)
            opt_bal = opt_bal + list(
                np.linspace(best_bal, init_bal, num_to_interpolate)
            )
            break
        else:
            bal_proposals = np.linspace(
                best_bal, min(init_bal, best_bal + 0.25), est_resolution
            )
    # interpolate the rest
    opt_bal = opt_bal[::-1]
    hist_metrics = {"a_s": hist_a_s, "hops": hist_hops}
    return x_cur, opt_bal, hist_metrics
