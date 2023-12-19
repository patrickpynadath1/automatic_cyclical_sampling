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
