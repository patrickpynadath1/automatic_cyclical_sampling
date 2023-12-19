import math
import torch
import torch.nn as nn
import torch.distributions as dists
import utils
import numpy as np


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
