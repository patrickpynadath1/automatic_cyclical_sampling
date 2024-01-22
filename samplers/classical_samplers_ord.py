import math
import torch
import torch.nn as nn
import torch.distributions as dists
import utils
import numpy as np


class PerDimGibbsSamplerOrd(nn.Module):
    def __init__(self, dim, max_val, rand=False):
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
        self.max_val = max_val

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
        # need to calculate the energies of all the possible values
        sample_expanded = torch.repeat_interleave(sample, self.max_val, dim=0)
        values_to_test = torch.Tensor([[i] for i in range(self.max_val)]).repeat(
            (sample.size(0), 1)
        )
        sample_expanded[:, self._i] = values_to_test[:, 0].to(sample.device)
        energies = model(sample_expanded).squeeze()
        cat_dist = dists.categorical.Categorical(
            energies.reshape((sample.size(0), self.max_val)).exp()
        )
        new_coords = cat_dist.sample()
        sample[:, self._i] = new_coords
        self._i = (self._i + 1) % self.dim
        self._hops = (x != sample).float().sum(-1).mean().item()
        self._ar = self._hops
        return sample

    def logp_accept(self, xhat, x, model):
        # only true if xhat was generated from self.step(x, model)
        return 0.0


class RandWalkOrd(nn.Module):
    def __init__(self, dim, max_val, rand=False):
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
        self.max_val = max_val

    def step(self, x, model):
        sample = x.clone()
        cat_dist = dists.categorical.Categorical(
            torch.zeros((sample.size(0), self.dim, self.max_val)).to(sample.device)
        )
        new_coords = cat_dist.sample()

        a = (la > torch.rand_like(la)).float()
        x = new_coords * a[:, None] + x * (1.0 - a[:, None])
        return x

    def logp_accept(self, xhat, x, model):
        # only true if xhat was generated from self.step(x, model)
        return 0.0


class PerDimMetropolisSamplerOrd(nn.Module):
    def __init__(self, dim, dist_to_test, max_val, rand=False):
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
        self.max_val = max_val
        self.dist_to_test = dist_to_test

    def step(self, x, model):
        if self.rand:
            i = np.random.randint(0, self.dim)
        else:
            i = self._i

        logits = []
        # ndim = x.size(-1)
        logits = torch.zeros((x.size(0), self.max_val)).to(x.device)
        len_values_to_test = 2 * self.dist_to_test + 1
        values_to_test = (
            torch.arange(-self.dist_to_test, self.dist_to_test + 1, step=1)[:, None]
            .repeat((x.size(0), 1))
            .to(x.device)
        )
        x_expanded = torch.repeat_interleave(x, len_values_to_test, dim=0)
        x_expanded[:, i] = torch.clamp(
            x_expanded[:, i] - values_to_test[:, 0], min=0, max=self.max_val - 1
        )
        coordinates_tested = x_expanded[:, i].reshape((x.size(0), len_values_to_test))
        energies = model(x_expanded).squeeze().reshape((x.size(0), len_values_to_test))
        logits[torch.arange(logits.size(0)).unsqueeze(1), coordinates_tested] = energies
        #
        #
        #
        # for k in range(-self.dist_to_test, self.dist_to_test + 1):
        #     sample = x.clone()
        #     # sample_i = torch.zeros((ndim,))
        #     # sample_i[k] = 1.0
        #     sample[:, i] = sample[:, i] + k
        #     # make sure values fall inside sample space
        #     sample[sample < 0] = 0
        #     sample[sample >= self.max_val] = self.max_val - 1
        #
        #     lp_k = model(sample).squeeze()
        #     logits[:, sample[:, self._i]] = lp_k
        dist = dists.Categorical(logits=logits)
        updates = dist.sample()
        sample = x.clone()
        sample[:, i] = updates
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
