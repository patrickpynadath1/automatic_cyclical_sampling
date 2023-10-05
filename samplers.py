import math

import torch
import torch.nn as nn
import torch.distributions as dists
import utils
import numpy as np
device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')

class LangevinSampler(nn.Module):
    def __init__(self, dim, n_steps=10, approx=False, multi_hop=False, fixed_proposal=False, temp=2., step_size=0.2, mh=True,
                 store_reject=False, store_diff=True):
        super().__init__()
        self.dim = dim
        self.n_steps = n_steps
        self._ar = 0.
        self._mt = 0.
        self._pt = 0.
        self._hops = 0.
        self._phops = 0.
        self.approx = approx
        self.fixed_proposal = fixed_proposal
        self.multi_hop = multi_hop
        self.temp = temp
        self.step_size = step_size
        self.initial_step_size = step_size
        self.store_reject = store_reject
        self.have_reject = False
        self.reject_sample = None
        self.store_diff = store_diff
        self.diff_values = []
        self.flip_probs = []
        if approx:
            self.diff_fn = lambda x, m: utils.approx_difference_function(x, m) / self.temp
        else:
            self.diff_fn = lambda x, m: utils.difference_function(x, m) / self.temp

        self.mh = mh
        self.a_s = []
        self.a_ss = []
        self.hops = []

    def get_name(self):
        if self.mh:
            base='dmala'
        else:
            base = 'dula'
        name = f'{base}_stepsize_{self.step_size}'
        return name


    def step(self, x, model):

        x_cur = x
        
        m_terms = []
        prop_terms = []
        
        EPS = 1e-10
        for i in range(self.n_steps):

            forward_delta = self.diff_fn(x_cur, model)
            if self.store_diff:
                self.diff_values.append(forward_delta.mean(dim=1).cpu().numpy())
            term2 = 1./(2*self.step_size) # for binary {0,1}, the L2 norm is always 1        
            flip_prob = torch.exp(forward_delta-term2)/(torch.exp(forward_delta-term2)+1)
            if self.store_diff:
                self.flip_probs.append(flip_prob.mean(dim=1).cpu().numpy())
            rr = torch.rand_like(x_cur)
            ind = (rr<flip_prob)*1
            x_delta = (1. - x_cur)*ind + x_cur * (1. - ind)

            if self.mh:
                probs = flip_prob*ind + (1 - flip_prob) * (1. - ind)
                lp_forward = torch.sum(torch.log(probs+EPS),dim=-1)

                reverse_delta = self.diff_fn(x_delta, model)
                flip_prob = torch.exp(reverse_delta-term2)/(torch.exp(reverse_delta-term2)+1)
                probs = flip_prob*ind + (1 - flip_prob) * (1. - ind)
                lp_reverse = torch.sum(torch.log(probs+EPS),dim=-1)
                
                m_term = (model(x_delta).squeeze() - model(x_cur).squeeze())
                la = m_term + lp_reverse - lp_forward
                a = (la.exp() > torch.rand_like(la)).float()
                a_mean = a.mean().item()
                self.a_s.append(a_mean)
                if self.store_reject:
                    if a_mean != 1:
                        self.have_reject = True
                        self.reject_sample = x_delta.detach()
                    else:
                        self.have_reject = False
                        self.reject_sample = None
                x_cur = x_delta * a[:, None] + x_cur * (1. - a[:, None])
            else:
                x_cur = x_delta

        return x_cur


class CyclicalLangevinSampler(nn.Module):
    def __init__(self,
                 dim,
                 device,
                 num_cycles,
                 num_iters,
                 n_steps=10,
                 approx=False,
                 multi_hop=False,
                 fixed_proposal=False,
                 temp=1.,
                 mean_stepsize=0.2,
                 mh=True,
                 initial_balancing_constant = 1,
                 use_balancing_constant=False,
                 include_exploration=False,
                 half_mh = False,
                 store_reject = False,
                 store_diff=False,
                 burn_in_adaptive=False,
                 adapt_rate = .05,
                 adapt_alg = None,
                 param_to_adapt = 'step'):
        super().__init__()
        self.device = device
        self.dim = dim
        self.n_steps = n_steps
        self._ar = 0.
        self._mt = 0.
        self._pt = 0.
        self._hops = 0.
        self._phops = 0.
        self.approx = approx
        self.fixed_proposal = fixed_proposal
        self.multi_hop = multi_hop
        self.temp = temp
        self.step_size = mean_stepsize
        self.initial_step_size = mean_stepsize
        self.num_cycles = num_cycles
        self.num_iters = num_iters
        self.use_balancing_constant = use_balancing_constant
        self.half_mh = half_mh
        self.store_reject = store_reject
        self.have_reject = False
        self.store_diff= store_diff
        self.reject_sample = None
        self.burn_in_adaptive = burn_in_adaptive
        self.adapt_rate = adapt_rate

        if approx:
            self.diff_fn = lambda x, m: utils.approx_difference_function(x, m)
        else:
            self.diff_fn = lambda x, m: utils.difference_function(x, m)
        self.mh = mh
        self.a_s = []
        self.hops = []
        self.initial_balancing_constant = initial_balancing_constant
        self.balancing_constant = initial_balancing_constant
        self.include_exploration = include_exploration

        self.iter_per_cycle = math.ceil(self.num_iters / self.num_cycles)
        # only using the exploitation stage -> twice as many iter per cycle
        self.step_sizes = self.calc_stepsizes(mean_stepsize)
        self.diff_values = []
        self.flip_probs = []
        if self.use_balancing_constant:
            self.balancing_constants = self.calc_balancing_constants(self.initial_balancing_constant)
        self.adapt_alg = adapt_alg
        self.param_to_adapt = param_to_adapt
        # need two arrays: one for explore step sizes, another for exploit step sizes


    def get_name(self):
        if self.mh:
            base='cyc_dmala'
        else:
            base = 'cyc_dula'
        name = f'{base}_cycles_{self.num_cycles}_stepsize_{self.initial_step_size}_usebal' \
               f'_{self.use_balancing_constant}_initbal_{self.balancing_constant}' \
               f'_include_exploration_{self.include_exploration}'
        if self.half_mh:
            name += "_halfMH"
        if self.burn_in_adaptive:
            name += f"_burnin_adaptive_{self.adapt_rate}_alg_{self.adapt_alg}_param_{self.param_to_adapt}"
        return name

    # i know there is a better way to code this
    # but i just want to see if this works

    def calc_stepsizes(self, mean_step):
        res = []
        total_iter = self.iter_per_cycle
        for k_iter in range(total_iter):
            inner = (np.pi * k_iter)/total_iter
            cur_step_size = mean_step * (np.cos(inner) + 1)
            step_size = cur_step_size
            res.append(step_size)
        res = torch.tensor(res, device=self.device)
        return res


    def calc_opt_acc_rate(self):
        res = []
        total_iter = self.iter_per_cycle
        a_0 = .148
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
            cur_balancing_constant = (init_bal - .5) / 2 * (np.cos(inner)) + (
                    init_bal + .5) / 2
            res.append(cur_balancing_constant)
        res = torch.tensor(res, device=self.device)
        return res

    def run_burnin_cycle_adaptive(self, x_init, model, adaptive_cycles, lr = .015, opt_acc = .574):
        # need to get optimal acceptance rate
        orig_mh = self.mh # saving it so we can reset it to normal after the adaptive cycle
        self.mh = True # setting it to be true
        x_cur = x_init
        param_values = []
        acc_rates = []
        for n in range(adaptive_cycles):
            mean_acc = 0
            for k in range(self.iter_per_cycle):
                x_delta, diff, p_acc = self.step(x_cur, model, k, return_diff=True)
                mean_acc += p_acc.detach().mean().item() / self.iter_per_cycle
                x_cur = x_delta
            if self.param_to_adapt == 'step':
                potential_adaptations = [self.step_size * (1 + lr * np.abs(mean_acc - opt_acc)),
                                         self.step_size * (1 - lr * np.abs(mean_acc - opt_acc))]
                orig = self.step_size
            elif self.param_to_adapt == 'bal':
                potential_adaptations = [min(self.balancing_constant * (1 + lr * np.abs(mean_acc - opt_acc)), 1),
                                         max(self.balancing_constant * (1 - lr * np.abs(mean_acc - opt_acc)), .5)]
                orig = self.balancing_constant

            potential_samples = [x_cur]
            probs = [mean_acc]
            for s in potential_adaptations:
                mean_acc = 0
                if self.param_to_adapt == 'step':
                    self.step_sizes = self.calc_stepsizes(s)
                elif self.param_to_adapt == 'bal':
                    self.balancing_constants = self.calc_balancing_constants(s)

                for k in range(self.iter_per_cycle):
                    x_delta, diff, p_acc = self.step(x_cur, model, k, return_diff=True)
                    mean_acc += p_acc.detach().mean().item() / self.iter_per_cycle
                    x_cur = x_delta
                potential_samples.append(x_cur)
                probs.append(mean_acc)
            i = np.argmin([np.abs(p - opt_acc) for p in probs])
            if i == 0:
                new_val = orig
            else:
                new_val = potential_adaptations[i-1]

            if self.param_to_adapt == 'step':
                self.step_sizes = self.calc_stepsizes(new_val)
                self.step_size = new_val
            elif self.param_to_adapt == 'bal':
                self.balancing_constants = self.calc_balancing_constants(new_val)
                self.balancing_constant = orig
            param_values.append(new_val)
            x_cur = potential_samples[i]
            # adapting the step size based on the mean
            acc_rates.append(probs[i])
        self.mh = orig_mh
        return param_values, np.array(acc_rates)

    def run_burnin_iter_adaptive(self, x_init, model, adaptive_cycles, lr = .015):
        # need to get optimal acceptance rate
        orig_mh = self.mh # saving it so we can reset it to normal after the adaptive cycle
        self.mh = True # setting it to be true
        opt_acc = self.calc_opt_acc_rate().cpu().numpy()
        x_cur = x_init
        param_values = []
        acc_rates = []
        for n in range(adaptive_cycles):
            tmp = []
            for k in range(self.iter_per_cycle):

                x_delta, diff, p_acc = self.step(x_cur, model, k, return_diff=True)
                p_mean = p_acc.detach().mean().item()
                target = opt_acc[k]
                change = lr * (p_mean - target)
                if self.param_to_adapt == 'step':
                    orig = self.step_sizes[k]
                    potential_changes = [orig * (1 - change),
                                         orig * (1 + change)]
                elif self.param_to_adapt == 'bal':
                    orig = self.balancing_constants[k]
                    potential_changes = [max(orig * (1 - change), .5),
                                         min(orig * (1 + change), 1)]
                new_probs = [p_mean]
                potential_samples = [x_delta]
                for s in potential_changes:
                    if self.param_to_adapt == 'step':
                        self.step_sizes[k] = s
                    elif self.param_to_adapt == 'bal':
                        self.balancing_constants[k] = s
                    x_delta, diff, p_acc = self.step(x_cur, model, k, return_diff=True)
                    new_probs.append(p_acc.detach().mean().item())
                    potential_samples.append(x_delta)
                i = np.argmin([np.abs(p - target) for p in new_probs])
                tmp.append(new_probs[i])
                x_cur = potential_samples[i]
                if i == 0:
                    new_val = orig
                else:
                    new_val = potential_changes[i-1]
                if self.param_to_adapt == 'step':
                    self.step_sizes[k] = new_val
                elif self.param_to_adapt == 'bal':
                    self.balancing_constants[k] = new_val
            if self.param_to_adapt == 'step':
                param_values.append(self.step_sizes.detach().cpu().numpy())
            elif self.param_to_adapt == 'bal':
                param_values.append(self.balancing_constants.detach().cpu().numpy())
            acc_rates.append(tmp)
        self.mh = orig_mh
        return param_values, np.array(acc_rates)


# TODO: finish implementing
    def run_burnin_fisher(self, x_init, model, adaptive_cycles, lam = .015):
        # need to get optimal acceptance rate
        orig_mh = self.mh # saving it so we can reset it to normal after the adaptive cycle
        self.mh = True # setting it to be true
        opt_acc = self.calc_opt_acc_rate()
        x_cur = x_init
        step_sizes = []
        for n in range(adaptive_cycles):
            for k in range(self.iter_per_cycle):
                x_delta, diff, lp_reverse = self.step(x_cur, model, k, return_diff=True)
                p_acc = torch.ones_like(lp_reverse, device=lp_reverse.device) - torch.exp(lp_reverse)
                s_k = torch.sqrt(p_acc) * diff
                if k == 1:
                    r_k = 1 / (1 + ((lam / (lam + torch.norm(s_k) ** 2)) ** .5))
                    g_k = (1 / lam ** .5) * (torch.ones_like(s_k, device=s_k.device) - r_k * (s_k ** 2)/(lam + torch.norm(s_k)))
                else:
                    phi_k = g_k * s_k
                    r_k = 1 / (1 + (1 + torch.norm(phi_k) ** 2) ** 5)
                    g_k = g_k - (r_k) * (g_k * phi_k ** 2)/(1 + torch.norm(phi_k) ** 2)
                x_cur = x_delta
            step_sizes.append(self.step_size)
        self.mh = orig_mh
        self.step_sizes = self.calc_stepsizes(self.step_size)
        return step_sizes



    def run_burnin_sun(self, x_init, model, adaptive_cycles):
        orig_mh = self.mh # saving it so we can reset it to normal after the adaptive cycle
        self.mh = True # setting it to be true
        param_values = []
        final_hops = []
        for i in range(adaptive_cycles):
            if self.param_to_adapt == 'step':
                orig = self.step_size
                potential_adaptions = [orig,
                                       orig*(1 - self.adapt_rate),
                                       orig * (1 + self.adapt_rate)]
            elif self.param_to_adapt == 'bal':
                orig = self.balancing_constant
                potential_adaptions = [orig,
                                       min(1, orig * (1 + self.adapt_rate)),
                                       max(.5, orig * (1 - self.adapt_rate))]


            hops = []
            for s in potential_adaptions:
                if self.param_to_adapt == 'step':
                    self.step_sizes = self.calc_stepsizes(s)
                elif self.param_to_adapt == 'bal':
                    self.balancing_constants = self.calc_balancing_constants(s)
                x_cur = x_init
                tmp_h = []
                for k in range(self.iter_per_cycle):
                    x_delta = self.step(x_cur, model, k)
                    cur_hops = (x_cur[0] != x_delta[0]).float().sum(-1).item()
                    tmp_h.append(cur_hops)
                    x_cur = x_delta
                hops.append(sum(tmp_h)/len(tmp_h))
            max_idx = np.argmax(hops)
            final_hops.append(hops[max_idx])
            new_val = potential_adaptions[max_idx]
            param_values.append(new_val)
            if self.param_to_adapt == 'step':
                self.step_size = new_val
                self.step_sizes = self.calc_stepsizes(self.step_size)
            elif self.param_to_adapt == 'bal':
                self.balancing_constant = new_val
                self.balancing_constants = self.calc_balancing_constants(new_val)
        self.mh = orig_mh
        return param_values, final_hops


    def step(self, x, model, k_iter,
             return_diff = False):

        x_cur = x

        m_terms = []
        prop_terms = []

        EPS = 1e-10
        step_size = self.step_sizes[k_iter % self.iter_per_cycle]
        if self.use_balancing_constant:
            balancing_constant = self.balancing_constants[k_iter % self.iter_per_cycle]
        else:
            balancing_constant = .5

        for i in range(self.n_steps):
            forward_delta= self.diff_fn(x_cur, model)
            forward_delta_bal = forward_delta * balancing_constant
            term2 = 1. / step_size  # for binary {0,1}, the L2 norm is always 1
            flip_prob = torch.exp(forward_delta_bal - term2) / (torch.exp(forward_delta_bal - term2) + 1)
            rr = torch.rand_like(x_cur)
            ind = (rr < flip_prob) * 1
            x_delta = (1. - x_cur) * ind + x_cur * (1. - ind)
            if self.mh:
                probs = flip_prob * ind + (1 - flip_prob) * (1. - ind)
                lp_forward = torch.sum(torch.log(probs + EPS), dim=-1)

                reverse_delta = self.diff_fn(x_delta, model) * balancing_constant
                flip_prob = torch.exp(reverse_delta - term2) / (torch.exp(reverse_delta - term2) + 1)
                probs = flip_prob * ind + (1 - flip_prob) * (1. - ind)
                lp_reverse = torch.sum(torch.log(probs + EPS), dim=-1)

                m_term = (model(x_delta).squeeze() - model(x_cur).squeeze())
                la = m_term + lp_reverse - lp_forward
                a = (la.exp() > torch.rand_like(la)).float()
                x_cur = x_delta * a[:, None] + x_cur * (1. - a[:, None])
            else:
                x_cur = x_delta
        if return_diff:
            probs = torch.minimum(torch.ones_like(la, device=la.device), la.exp())
            return x_cur, forward_delta, probs
        return x_cur


#

# Gibbs-With-Gradients for binary data
class DiffSampler(nn.Module):
    def __init__(self, dim, n_steps=10, approx=False, multi_hop=False, fixed_proposal=False, temp=2., step_size=1.0):
        super().__init__()
        self.dim = dim
        self.n_steps = n_steps
        self._ar = 0.
        self._mt = 0.
        self._pt = 0.
        self._hops = 0.
        self._phops = 0.
        self.approx = approx
        self.fixed_proposal = fixed_proposal
        self.multi_hop = multi_hop
        self.temp = temp
        self.step_size = step_size
        if approx:
            self.diff_fn = lambda x, m: utils.approx_difference_function(x, m) / self.temp
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
                    x_delta = (1. - x_cur) * changes + x_cur * (1. - changes)
                    la = (model(x_delta).squeeze() - model(x_cur).squeeze())
                    a = (la.exp() > torch.rand_like(la)).float()
                    x_cur = x_delta * a[:, None] + x_cur * (1. - a[:, None])
                    a_s.append(a.mean().item())
                self._ar = np.mean(a_s)
            else:
                for i in range(self.n_steps):
                    forward_delta = self.diff_fn(x_cur, model)
                    cd_forward = dists.Bernoulli(logits=(forward_delta * 2 / self.temp))
                    changes = cd_forward.sample()

                    lp_forward = cd_forward.log_prob(changes).sum(-1)

                    x_delta = (1. - x_cur) * changes + x_cur * (1. - changes)


                    reverse_delta = self.diff_fn(x_delta, model)
                    cd_reverse = dists.Bernoulli(logits=(reverse_delta * 2 / self.temp))

                    lp_reverse = cd_reverse.log_prob(changes).sum(-1)

                    m_term = (model(x_delta).squeeze() - model(x_cur).squeeze())
                    la = m_term + lp_reverse - lp_forward
                    a = (la.exp() > torch.rand_like(la)).float()
                    x_cur = x_delta * a[:, None] + x_cur * (1. - a[:, None])
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

                    x_delta = (1. - x_cur) * changes + x_cur * (1. - changes)
                    la = (model(x_delta).squeeze() - model(x_cur).squeeze())
                    a = (la.exp() > torch.rand_like(la)).float()
                    x_cur = x_delta * a[:, None] + x_cur * (1. - a[:, None])
                    a_s.append(a.mean().item())
                self._ar = np.mean(a_s)
            else:
                for i in range(self.n_steps):
                    forward_delta = self.diff_fn(x_cur, model)
                    cd_forward = dists.OneHotCategorical(logits=forward_delta)
                    changes = cd_forward.sample()

                    lp_forward = cd_forward.log_prob(changes)

                    x_delta = (1. - x_cur) * changes + x_cur * (1. - changes)

                    reverse_delta = self.diff_fn(x_delta, model)
                    cd_reverse = dists.OneHotCategorical(logits=reverse_delta)

                    lp_reverse = cd_reverse.log_prob(changes)

                    m_term = (model(x_delta).squeeze() - model(x_cur).squeeze())
                    la = m_term + lp_reverse - lp_forward
                    a = (la.exp() > torch.rand_like(la)).float()
                    x_cur = x_delta * a[:, None] + x_cur * (1. - a[:, None])

        return x_cur


# Gibbs-With-Gradients variant which proposes multiple flips per step
class MultiDiffSampler(nn.Module):
    def __init__(self, dim, n_steps=10, approx=False, temp=1., n_samples=1):
        super().__init__()
        self.dim = dim
        self.n_steps = n_steps
        self._ar = 0.
        self._mt = 0.
        self._pt = 0.
        self._hops = 0.
        self._phops = 0.
        self.approx = approx
        self.temp = temp
        self.n_samples = n_samples
        if approx:
            self.diff_fn = lambda x, m: utils.approx_difference_function(x, m) / self.temp
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

            changes = (changes_all.sum(0) > 0.).float()

            x_delta = (1. - x_cur) * changes + x_cur * (1. - changes)
            # self._phops = (x_delta != x).float().sum(-1).mean().item()
            cur_hops = (x_cur[0] != x_delta[0]).float().sum(-1).item()
            self.hops.append(cur_hops)

            reverse_delta = self.diff_fn(x_delta, model)
            cd_reverse = dists.OneHotCategorical(logits=reverse_delta)

            lp_reverse = cd_reverse.log_prob(changes_all).sum(0)

            m_term = (model(x_delta).squeeze() - model(x_cur).squeeze())
            la = m_term + lp_reverse - lp_forward
            a = (la.exp() > torch.rand_like(la)).float()
            x_cur = x_delta * a[:, None] + x_cur * (1. - a[:, None])
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
        self.change_rate = 0.
        self.p = nn.Parameter(torch.zeros((dim,)))
        self._i = 0
        self._ar = 0.
        self._hops = 0.
        self._phops = 1.
        self.rand = rand

    def step(self, x, model):
        sample = x.clone()
        lp_keep = model(sample).squeeze()
        if self.rand:
            changes = dists.OneHotCategorical(logits=torch.zeros((self.dim,))).sample((x.size(0),)).to(x.device)
        else:
            changes = torch.zeros((x.size(0), self.dim)).to(x.device)
            changes[:, self._i] = 1.

        sample_change = (1. - changes) * sample + changes * (1. - sample)

        lp_change = model(sample_change).squeeze()

        lp_update = lp_change - lp_keep
        update_dist = dists.Bernoulli(logits=lp_update)
        updates = update_dist.sample()
        sample = sample_change * updates[:, None] + sample * (1. - updates[:, None])
        self.changes[self._i] = updates.mean()
        self._i = (self._i + 1) % self.dim
        self._hops = (x != sample).float().sum(-1).mean().item()
        self._ar = self._hops
        return sample

    def logp_accept(self, xhat, x, model):
        # only true if xhat was generated from self.step(x, model)
        return 0.

class PerDimMetropolisSampler(nn.Module):
    def __init__(self, dim, n_out, rand=False):
        super().__init__()
        self.dim = dim
        self.n_out = n_out
        self.changes = torch.zeros((dim,))
        self.change_rate = 0.
        self.p = nn.Parameter(torch.zeros((dim,)))
        self._i = 0
        self._j = 0
        self._ar = 0.
        self._hops = 0.
        self._phops = 0.
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
            sample_i[k] = 1.
            sample[:, i, :] = sample_i
            lp_k = model(sample).squeeze()
            logits.append(lp_k[:, None])
        logits = torch.cat(logits, 1)
        dist = dists.OneHotCategorical(logits=logits)
        updates = dist.sample()
        sample = x.clone()
        sample[:, i, :] = updates
        self._i = (self._i + 1) % self.dim
        self._hops = ((x != sample).float().sum(-1) / 2.).sum(-1).mean().item()
        self._ar = self._hops
        return sample

    def logp_accept(self, xhat, x, model):
        # only true if xhat was generated from self.step(x, model)
        return 0.

class PerDimLB(nn.Module):
    def __init__(self, dim, rand=False):
        super().__init__()
        self.dim = dim
        self.changes = torch.zeros((dim,))
        self.change_rate = 0.
        self.p = nn.Parameter(torch.zeros((dim,)))
        self._i = 0
        self._j = 0
        self._ar = 0.
        self._hops = 0.
        self._phops = 0.
        self.rand = rand

    def step(self, x, model):
        logits = []
        ndim = x.size(-1)
        fx = model(x).squeeze()
        for k in range(ndim):
            sample = x.clone()
            sample[:, k] = 1-sample[:, k] 
            lp_k = (model(sample).squeeze()-fx)/2.
            logits.append(lp_k[:, None])
        logits = torch.cat(logits, 1)
        Z_forward = torch.sum(torch.exp(logits),dim=-1)
        dist = dists.OneHotCategorical(logits=logits)
        changes = dist.sample()
        x_delta = (1. - x) * changes + x * (1. - changes)
        fx_delta = model(x_delta)
        logits = []
        for k in range(ndim):
            sample = x_delta.clone()
            sample[:, k] = 1-sample[:, k] 
            lp_k = (model(sample).squeeze()-fx_delta)/2.
            logits.append(lp_k[:, None])
        logits = torch.cat(logits, 1)
        Z_reverse = torch.sum(torch.exp(logits),dim=-1)
        la =  Z_forward/Z_reverse
        a = (la > torch.rand_like(la)).float()
        x = x_delta * a[:, None] + x * (1. - a[:, None])
        # a_s.append(a.mean().item())
        # self._ar = np.mean(a_s)
        return x

    def logp_accept(self, xhat, x, model):
        # only true if xhat was generated from self.step(x, model)
        return 0.


# Gibbs-With-Gradients for categorical data
class DiffSamplerMultiDim(nn.Module):
    def __init__(self, dim, n_steps=10, approx=False, temp=1.):
        super().__init__()
        self.dim = dim
        self.n_steps = n_steps
        self._ar = 0.
        self._mt = 0.
        self._pt = 0.
        self._hops = 0.
        self._phops = 0.
        self.approx = approx
        self.temp = temp
        if approx:
            self.diff_fn = lambda x, m: utils.approx_difference_function_multi_dim(x, m) / self.temp
        else:
            self.diff_fn = lambda x, m: utils.difference_function_multi_dim(x, m) / self.temp

    def step(self, x, model):

        x_cur = x
        a_s = []
        m_terms = []
        prop_terms = []


        for i in range(self.n_steps):
            constant = 1.
            forward_delta = self.diff_fn(x_cur, model)
            
            # make sure we dont choose to stay where we are!
            forward_logits = forward_delta - constant * x_cur
            #print(forward_logits)
            cd_forward = dists.OneHotCategorical(logits=forward_logits.view(x_cur.size(0), -1))
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
            x_delta = x_cur.clone() * (1. - changed_ind[:, :, None]) + changes_r

            reverse_delta = self.diff_fn(x_delta, model)
            reverse_logits = reverse_delta - constant * x_delta
            cd_reverse = dists.OneHotCategorical(logits=reverse_logits.view(x_delta.size(0), -1))
            reverse_changes = x_cur * changed_ind[:, :, None]

            lp_reverse = cd_reverse.log_prob(reverse_changes.view(x_delta.size(0), -1))

            m_term = (model(x_delta).squeeze() - model(x_cur).squeeze())
            la = m_term + lp_reverse - lp_forward
            a = (la.exp() > torch.rand_like(la)).float()
            x_cur = x_delta * a[:, None, None] + x_cur * (1. - a[:, None, None])
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
        self.change_rate = 0.
        self.p = nn.Parameter(torch.zeros((dim,)))

    def step(self, x, model):
        sample = x.clone()
        for i in range(self.dim):
            lp_keep = model(sample).squeeze()

            xi_keep = sample[:, i]
            xi_change = 1. - xi_keep
            sample_change = sample.clone()
            sample_change[:, i] = xi_change

            lp_change = model(sample_change).squeeze()

            lp_update = lp_change - lp_keep
            update_dist = dists.Bernoulli(logits=lp_update)
            updates = update_dist.sample()
            sample = sample_change * updates[:, None] + sample * (1. - updates[:, None])
            self.changes[i] = updates.mean()
        return sample

    def logp_accept(self, xhat, x, model):
        # only true if xhat was generated from self.step(x, model)
        return 0.
