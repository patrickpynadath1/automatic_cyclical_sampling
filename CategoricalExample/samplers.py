import torch
import torch.nn as nn
import torch.distributions as dists
import numpy as np


class LangevinSamplerMultiDim(nn.Module):
    def __init__(
        self,
        dim,
        num_cls=3,
        n_steps=10,
        multi_hop=False,
        temp=1.0,
        bal_constant=0.5,
        step_size=0.2,
        mh=True,
        device=None,
    ):
        super().__init__()
        self.dim = dim
        self.n_steps = n_steps
        self._ar = 0.0
        self._mt = 0.0
        self._pt = 0.0
        self._hops = 0.0
        self._phops = 0.0
        self.multi_hop = multi_hop
        self.temp = temp
        self.step_size = step_size  # rbm sampling: accpt prob is about 0.5 with lr = 0.2, update 16 dims per step (total 784 dims). ising sampling: accept prob 0.5 with lr=0.2
        # ising learning: accept prob=0.7 with lr=0.2
        # ebm: statistic mnist: accept prob=0.45 with lr=0.2
        self.a_s = []
        self.mh = mh
        self.bal = bal_constant
        self.num_cls = num_cls  ### number of classes in each dimension

    def get_grad(self, x, model):
        x = x.requires_grad_()
        out = model(x)
        gx = torch.autograd.grad(out.sum(), x)[0]
        return gx.detach() * self.bal

    def to_one_hot(self, x):
        x_one_hot = torch.zeros((x.shape[0], self.dim, self.num_cls)).to(x.device)
        x_one_hot[:, range(self.dim), x[0, :]] = 1.0

        return x_one_hot

    def step(self, x, model):
        """
        input x : bs * dim, every dim contains a integer of 0 to (num_cls-1)
        """
        x_cur = x
        m_terms = []
        prop_terms = []

        EPS = 1e-10
        for i in range(self.n_steps):
            x_cur_one_hot = self.to_one_hot(x_cur.long())
            grad = self.get_grad(x_cur_one_hot, model) / self.temp

            ### we are going to create first term: bs * dim * num_cls, second term: bs * dim * num_cls
            grad_cur = grad[0:1, range(self.dim), x_cur[0, :]]
            first_term = grad.detach().clone() - grad_cur.unsqueeze(2).repeat(
                1, 1, self.num_cls
            )

            second_term = torch.ones_like(first_term).to(x_cur.device) / (
                2 * self.step_size
            )
            second_term[0, range(self.dim), x_cur[0, :]] = 0.0
            logits = first_term - second_term
            cat_dist = torch.distributions.categorical.Categorical(logits=logits)
            x_delta = cat_dist.sample()

            if self.mh:
                lp_forward = torch.sum(cat_dist.log_prob(x_delta), dim=1)
                x_delta_one_hot = self.to_one_hot(x_delta)
                grad_delta = self.get_grad(x_delta_one_hot, model) / self.temp

                grad_delta_cur = grad[0:1, range(self.dim), x_delta[0, :]]
                first_term_delta = (
                    grad_delta.detach().clone()
                    - grad_delta_cur.unsqueeze(2).repeat(1, 1, self.num_cls)
                )

                second_term_delta = (
                    torch.ones_like(first_term_delta).to(x_delta.device)
                    / self.step_size
                )
                second_term_delta[0, range(self.dim), x_delta[0, :]] = 0.0

                cat_dist_delta = torch.distributions.categorical.Categorical(
                    logits=first_term_delta - second_term_delta
                )
                lp_reverse = torch.sum(cat_dist_delta.log_prob(x_cur), dim=1)

                m_term = (
                    model(x_delta_one_hot).squeeze() - model(x_cur_one_hot).squeeze()
                )
                la = m_term + lp_reverse - lp_forward
                a = (la.exp() > torch.rand_like(la)).float()
                self.a_s.append(a.mean().item())
                x_cur = x_delta * a[:, None] + x_cur * (1.0 - a[:, None])
            else:
                x_cur = x_delta
            x_cur = x_cur.long()
        return x_cur


# DLP but for ordinal values
class LangevinSamplerOrdinal(nn.Module):
    def __init__(
        self,
        dim,
        max_val=3,
        n_steps=10,
        multi_hop=False,
        temp=2.0,
        step_size=0.2,
        mh=True,
        device=None,
    ):
        super().__init__()
        self.dim = dim
        self.n_steps = n_steps
        self._ar = 0.0
        self._mt = 0.0
        self._pt = 0.0
        self._hops = 0.0
        self._phops = 0.0
        self.multi_hop = multi_hop
        self.temp = temp
        self.step_size = step_size  # rbm sampling: accpt prob is about 0.5 with lr = 0.2, update 16 dims per step (total 784 dims). ising sampling: accept prob 0.5 with lr=0.2
        # ising learning: accept prob=0.7 with lr=0.2
        # ebm: statistic mnist: accept prob=0.45 with lr=0.2
        self.a_s = []
        self.mh = mh
        self.max_val = max_val  ### number of classes in each dimension

    def get_grad(self, x, model):
        x = x.requires_grad_()
        out = model(x)
        gx = torch.autograd.grad(out.sum(), x)[0]
        return gx.detach()

    def _calc_logits(self, x_cur, grad):
        # creating the tensor of discrete values to compute the probabilities for
        batch_size = x_cur.shape[0]
        disc_values = torch.tensor([i for i in range(self.max_val)])[None, None, :]
        disc_values = disc_values.repeat((batch_size, self.dim, 1)).to(x_cur.device)
        term1 = torch.zeros((batch_size, self.dim, self.max_val))
        term2 = torch.zeros((batch_size, self.dim, self.max_val))
        x_expanded = x_cur[:, :, None].repeat((1, 1, 64)).to(x_cur.device)
        grad_expanded = grad[:, :, None].repeat((1, 1, 64)).to(x_cur.device)
        term1 = grad_expanded * (disc_values - x_expanded)
        term2 = (disc_values - x_expanded) ** 2 * (1 / (2 * self.step_size))
        return term1 - term2
        # for i in range(self.max_val):
        #     term1[:, :, i] = grad * (disc_values[:, :, i] - x_cur)
        #     term2[:, :, i] = (
        #         (disc_values[:, :, i] - x_cur) ** 2 * 1 / (2 * self.step_size)
        #     )
        # return term1 - term2

    def step(self, x, model):
        """
        input x : bs * dim, every dim contains a integer of 0 to (num_cls-1)
        """
        x_cur = x
        m_terms = []
        prop_terms = []

        EPS = 1e-10
        for i in range(self.n_steps):
            # batch size X dim
            grad = self.get_grad(x_cur.float(), model)
            logits = self._calc_logits(x_cur, grad)
            cat_dist = torch.distributions.categorical.Categorical(logits=logits)
            x_delta = cat_dist.sample()

            if self.mh:
                lp_forward = torch.sum(cat_dist.log_prob(x_delta), dim=1)
                grad_delta = self.get_grad(x_delta.float(), model) / self.temp

                logits_reverse = self._calc_logits(x_delta, grad_delta)

                cat_dist_delta = torch.distributions.categorical.Categorical(
                    logits=logits_reverse
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


class CycLangevinSamplerMultiDim(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def step(self, x, model, k_iter):
        return
