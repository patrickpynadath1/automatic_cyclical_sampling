import math
import torch
import torch.nn as nn
import torch.distributions as dists
import utils
import numpy as np
from .tuning_components import estimate_alpha_max

device = torch.device("cuda:" + str(0) if torch.cuda.is_available() else "cpu")


class MidLangevinSampler(nn.Module):
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
        self.D = None
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
        self.mid_grad = None
        self.calc_mid = False
        self.mh = mh
        self.a_s = []
        self.hops = []
        self.track_grad = False 
        self.t1 = None

    def get_name(self):

        return "mid"
    
    def mid_diff(self, x, model):
        # calculate the compliment of x 
        delta = self.mid_grad * -(2.0 * x - 1)
        return delta

    def step(self, x, model, use_dula=False):

        if not self.calc_mid: 
            mid_point = torch.ones_like(x) * .5 
            mid_point.requires_grad_()
            mid_grad = torch.autograd.grad(model(mid_point).sum(), mid_point)[0]
            self.mid_grad = mid_grad
            self.calc_mid = True
    
        x_cur = x

        m_terms = []
        prop_terms = []

        EPS = 1e-10
        for i in range(self.n_steps):
            forward_delta = self.mid_diff(x_cur, model) * .5
            if self.D is not None: 
                term2 = self.D * (1.0 / (2 * self.step_size))
            else: 
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

                reverse_delta = self.mid_diff(x_delta, model) * .5 
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
        self.D = None
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
        self.track_terms = True 
        self.t1 = None
        self.counts = 0

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
            if self.D is not None: 
                term2 = self.D * (1.0 / (2 * self.step_size))
            else: 
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
            # keeping track of forward deltas for the diagonal correction term 
            if self.t1 is None: 
                self.t1 = (forward_delta.max() - forward_delta.min()).item()
            else: 
                self.t1 = max((forward_delta.max() - forward_delta.min()).item(), self.t1)
            self.counts += 1
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
        return x_cur.detach()


# class TiledLangevinSampler(nn.Module):
#     def __init__(
#         self,
#         dim,
#         n_steps=10,
#         approx=False,
#         multi_hop=False,
#         fixed_proposal=False,
#         temp=1.0,
#         step_size=0.2,
#         mh=True,
#         bal=0.5,
#         n_tiles = 4
# #     ):
# #         super().__init__()
# #         self.dim = dim
# #         self.n_steps = n_steps
# #         self._ar = 0.0
# #         self._mt = 0.0
# #         self._pt = 0.0
# #         self._hops = 0.0
# #         self._phops = 0.0
# #         self.approx = approx
# #         self.fixed_proposal = fixed_proposal
# #         self.multi_hop = multi_hop
# #         self.temp = temp
# #         self.step_size = step_size
# #         self.bal = bal
# #         self.D = None
# #         if approx:
# #             self.diff_fn = (
# #                 lambda x, m: self.bal
# #                 * utils.approx_difference_function(x, m)
# #                 / self.temp
# #             )
# #         else:
# #             self.diff_fn = (
# #                 lambda x, m: self.bal * utils.difference_function(x, m) / self.temp
# #             )

# #         self.mh = mh
# #         self.a_s = []
# #         self.hops = []

# #         # need to calculate the tile indices  
# #         chunk_size = dim // n_tiles
# #         self.indices = torch.split(torch.arange(start=0, end =dim), int(chunk_size))

# #         # below code is for when sampling directly in image space
# #         # axis_indices = torch.split(torch.arange(start=0, end=dim), chunk_size)
# #         # self.tile_indices = []
# #         # for x_idx in axis_indices: 
# #         #     for y_idx in axis_indices: 
# #         #         chunk_indices = torch.cartesian_prod(x_idx, y_idx)
# #         #         self.tile_indices.append(chunk_indices)
        


#     def get_name(self):
#         if self.mh:
#             base = "dmala"

#         else:
#             base = "dula"
#         name = f"{base}_stepsize_{self.step_size}"

#         return name


#     def step(self, x, model, step_itr, use_dula=False,):
#         x_cur = x

#         m_terms = []
#         prop_terms = []

#         EPS = 1e-10
#         for i in range(self.n_steps):
#             term2 = 1.0 / (
#                 2 * self.step_size
#             )  # for binary {0,1}, the L2 norm is always 1
#             # idx_to_change = self.indices[step_itr % len(self.indices)]
#             forward_delta = self.diff_fn(x_cur, model)
#             idx_to_change = torch.argsort(forward_delta, dim=-1, descending=True)
#             idx_to_change = idx_to_change[:, :40]
            
#             forward_delta = torch.take(forward_delta, idx_to_change)
#             flip_prob_idx = torch.exp(forward_delta - term2) / (
#                 torch.exp(forward_delta - term2) + 1
#             )
#             rr = torch.rand_like(flip_prob_idx)
#             x_cur_idx = torch.take(x_cur, idx_to_change)
#             ind = (rr < flip_prob_idx) * 1
#             x_delta_idx = (1.0 - x_cur_idx) * ind + x_cur_idx * (1.0 - ind)
#             x_delta = x_cur.clone()
#             x_delta[:, idx_to_change] = x_delta_idx
#             if self.mh:
#                 probs = flip_prob * ind + (1 - flip_prob) * (1.0 - ind)
#                 lp_forward = torch.sum(torch.log(probs + EPS)[:, idx_to_change], dim=-1)

#                 reverse_delta = self.diff_fn(x_delta, model) 
#                 flip_prob = torch.exp(reverse_delta - term2) / (
#                     torch.exp(reverse_delta - term2) + 1
#                 )
#                 probs = flip_prob * ind + (1 - flip_prob) * (1.0 - ind)
#                 lp_reverse = torch.sum(torch.log(probs + EPS), dim=-1)

#                 m_term = model(x_delta).squeeze() - model(x_cur).squeeze()
#                 la = m_term + lp_reverse - lp_forward
#                 a = (la.exp() > torch.rand_like(la)).float()
#                 # self.a_s.append(a.mean().item())
#                 probs_tmp = torch.minimum(
#                     torch.ones_like(la, device=la.device), la.exp()
#                 )
#                 self.a_s.append(probs_tmp.detach().mean().item())
#                 if use_dula:
#                     x_cur = x_delta
#                 else:
#                     x_cur = x_delta * a[:, None] + x_cur * (1.0 - a[:, None])
#             else:
#                 x_cur = x_delta
#         return x_cur


