import torch
import torch.nn as nn
import numpy as np


# DLP but for ordinal values
class LangevinSamplerOrdinal(nn.Module):
    def __init__(
        self,
        dim,
        bal,
        max_val=3,
        n_steps=10,
        multi_hop=False,
        temp=1.0,
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
        # rbm sampling: accpt prob is about 0.5 with lr = 0.2, update 16 dims per step (total 784 dims). ising sampling: accept prob 0.5 with lr=0.2
        # ising learning: accept prob=0.7 with lr=0.2
        # ebm: statistic mnist: accept prob=0.45 with lr=0.2
        self.a_s = []
        self.bal = bal
        self.mh = mh
        self.max_val = max_val  ### number of classes in each dimension
        self.step_size = (step_size * self.max_val) ** (self.dim**2)
        # self.step_size = step_size

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
        x_expanded = x_cur[:, :, None].repeat((1, 1, self.max_val)).to(x_cur.device)
        grad_expanded = grad[:, :, None].repeat((1, 1, self.max_val)).to(x_cur.device)
        term1 = self.bal * grad_expanded * (disc_values - x_expanded)
        term2 = (disc_values - x_expanded) ** 2 * (1 / (2 * self.step_size))
        return term1 - term2

    def step(self, x, model, use_dula=False):
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
                if use_dula:
                    x_cur = x_delta
                else:
                    x_cur = x_delta * a[:, None] + x_cur * (1.0 - a[:, None])
            else:
                x_cur = x_delta
            x_cur = x_cur.long()
        return x_cur
