import math
import torch
import torch.nn as nn
import utils
from .tuning_components import *
from .dlp_samplers import LangevinSampler


class AutomaticCyclicalSampler(nn.Module):
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
        if iter_per_cycle and sbc:
            self.iter_per_cycle = iter_per_cycle
        else:
            print(self.num_iters)
            print(self.num_cycles)
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
            base = "acs"
        else:
            base = "acs_no_mh"
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
            if self.burnin_adaptive:
                name = base + f"_adaptive_lr_{self.burnin_lr}_a_s_cut_{self.a_s_cut}"
            else:
                name = (
                    base
                    + f"_cycle_length_{self.iter_per_cycle}_big_s_{self.big_step}_b_{self.big_bal}_small_s_{self.small_step}_b_{self.small_bal}"
                )
        if self.min_lr:
            name += f"_min_lr_{self.min_lr}"
        return name

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

    def adapt_big_step(
        self,
        x_init,
        model,
        budget,
        init_big_step,
        init_big_bal,
        lr,
        test_steps,
        a_s_cut,
        use_dula,
        bdmala=None,
    ):
        if bdmala is None:
            bdmala = LangevinSampler(
                dim=self.dim,
                n_steps=self.n_steps,
                approx=self.approx,
                multi_hop=self.multi_hop,
                fixed_proposal=self.fixed_proposal,
                step_size=1,
                mh=True,
                bal=init_big_bal,
            )

        (
            x_cur,
            alpha_max,
            alpha_max_metrics,
            _,
        ) = estimate_alpha_max(
            model=model,
            bdmala=bdmala,
            a_s_cut=a_s_cut,
            init_bal=init_big_bal,
            test_steps=test_steps,
            budget=budget,
            init_step_size=init_big_step,
            x_init=x_init,
            use_dula=use_dula,
            lr=lr,
        )
        self.step_sizes[0] = alpha_max
        self.balancing_constants[0] = init_big_bal
        return x_cur, alpha_max, alpha_max_metrics

    def adapt_small_step(
        self,
        x_init,
        model,
        budget,
        init_small_step,
        init_small_bal,
        lr,
        test_steps,
        a_s_cut,
        use_dula,
        bdmala=None,
    ):
        if bdmala is None:
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
        x_cur, alpha_min, alpha_min_metrics, _ = estimate_alpha_min(
            model=model,
            bdmala=bdmala,
            x_cur=x_init,
            budget=budget,
            init_step_size=init_small_step,
            test_steps=test_steps,
            lr=lr,
            a_s_cut=a_s_cut,
            init_bal=init_small_bal,
        )
        for i in range(1, len(self.step_sizes)):
            self.step_sizes[i] = alpha_min
            self.balancing_constants[i] = init_small_bal
        return x_cur, alpha_min, alpha_min_metrics

    def tuning_alg(
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
        use_bal_cyc=False,
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
        for i in range(50):
            dula_x_init = bdmala.step(dula_x_init, model).detach()
        bdmala.mh = True
        total_res = {}
        possible_x_inits = [dula_x_init]
        # estimating alpha min
        use_dula = not self.mh
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
        )
        possible_x_inits.append(alpha_min_x_init)
        (alpha_max_x_init, alpha_max, alpha_max_metrics, itrr) = estimate_alpha_max(
            model=model,
            bdmala=bdmala,
            a_s_cut=a_s_cut,
            init_bal=init_big_bal,
            test_steps=test_steps,
            budget=budget // 2,
            init_step_size=init_big_step,
            x_init=alpha_min_x_init,
            use_dula=use_dula,
        )

        init_big_bal = bdmala.bal
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
        return possible_x_inits[-1], total_res

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
            flip_prob = torch.exp(
                forward_delta_bal - term2
            ) / (
                torch.exp(forward_delta_bal - term2)
                + 1
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
                    torch.exp(reverse_delta - term2)
                    + 1
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
