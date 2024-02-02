import math
import torch
import torch.nn as nn
import torch.distributions as dists
import utils
import numpy as np
from .tuning_components import estimate_alpha_max

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

    def get_name(self):
        if self.mh:
            base = "dmala"

        else:
            base = "dula"
        name = f"{base}_stepsize_{self.step_size}"

        return name

    def adapt_big_step(
        self,
        x_cur,
        model,
        budget,
        test_steps,
        lr,
        init_big_step,
        a_s_cut,
        init_bal,
        use_dula,
    ):
        orig_mh = self.mh
        self.bal = init_bal
        for i in range(100):
            x_cur = self.step(x_cur.detach(), model)
        self.mh = orig_mh
        (
            x_cur,
            alpha_max,
            alpha_max_metrics,
            _,
        ) = estimate_alpha_max(
            model=model,
            bdmala=self,
            a_s_cut=a_s_cut,
            init_bal=init_bal,
            test_steps=test_steps,
            budget=budget,
            init_step_size=init_big_step,
            x_init=x_cur,
            use_dula=use_dula,
            lr=lr,
        )
        self.step_size = alpha_max
        self.bal = init_bal
        return x_cur, alpha_max, alpha_max_metrics

    def step(self, x, model, use_dula=False):
        x_cur = x

        m_terms = []
        prop_terms = []

        EPS = 1e-10
        for i in range(self.n_steps):
            forward_delta = self.diff_fn(x_cur, model)
            term2 = 1.0 / (
                2 * self.step_size
            )  # for binary {0,1}, the L2 norm is always 1
            flip_prob = torch.exp(forward_delta - term2) / (
                torch.exp(forward_delta - term2) + 1
            )
            orig_flip_prob = flip_prob
            rr = torch.rand_like(x_cur)
            ind = (rr < flip_prob) * 1
            orig_ind = ind
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
                if use_dula:
                    x_cur = x_delta
                else:
                    x_cur = x_delta * a[:, None] + x_cur * (1.0 - a[:, None])
            else:
                x_cur = x_delta
        return x_cur
