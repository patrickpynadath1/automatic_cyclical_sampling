from .acs_samplers import AutomaticCyclicalSampler
import torch
import torch.nn as nn
from .tuning_components import *

class AutomaticCyclicalSamplerOneHot(AutomaticCyclicalSampler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.unique_in_batch = False
        self.vocab_size = kwargs['vocab_size']
        self.embeddings = kwargs["embeddings"]
        self.proposal_temp = kwargs["proposal_temp"]
        self.is_acs = kwargs["is_acs"]
        self.a_s = []
        # what mass is placed on the top proposed token? 
        self.metrics = {
            'top_mass': [], 
            'a_s': [],
            'proposed_energy': [],
            'orig_energy': [],
            'hops':[],
            'mse_jump': [] # for embeds, compute the mse between embeds 
        }

    def calc_logits(self, tokens, grad, indices):
        if self.unique_in_batch:  
            batch_idx = torch.arange(indices.shape[0]).unsqueeze(-1)
            tokens = tokens[batch_idx, indices]
            grad = grad[batch_idx, indices]
        else: 
            tokens = tokens[:, indices]
            grad = grad[:, indices, :]

        # calculating term 1 
        grad_cur = torch.gather(grad, dim=-1, index=tokens.unsqueeze(-1)).squeeze() 
        first_term = grad.detach().clone() - grad_cur.unsqueeze(-1).repeat(1, 1, self.vocab_size) 
             
        # second_term = torch.ones_like(first_term).to(x_cur.device) / self.step_size
        # second_term[x_cur.unsqueeze(-1)] = 0.
        second_term = torch.abs(torch.nn.functional.one_hot(tokens, self.vocab_size) - 1) * 1. 
        return first_term, second_term 
    

    def step(self, theta, model, step_idx, **kwargs):
        if self.is_acs:
            step_size = self.step_sizes[step_idx % len(self.step_sizes)]
            balancing_constant = self.balancing_constants[step_idx % len(self.balancing_constants)]
        else: 
            step_size = self.step_size 
            balancing_constant = self.bal

        theta_tokens = theta['input_ids']
        theta_prime_tokens = theta_tokens.clone()
        theta_onehot = theta['inputs']
        masked_indices = theta['masked_indices']
        attention_mask = theta['attention_mask']

        grad = self._calc_grad(model, theta_onehot, attention_mask, masked_indices)
        t1, t2 = self.calc_logits(theta_tokens, grad, masked_indices)
        theta_forward_logits = balancing_constant * t1 - t2 / (2 * step_size)
        theta_forward_dist = torch.distributions.Categorical(logits=theta_forward_logits / self.proposal_temp)
        self.metrics['top_mass'].append(theta_forward_logits.softmax(dim=-1).max(dim=-1).values.cpu().tolist())
        # theta_prime_tokens[:, masked_indices] = theta_forward_dist.sample()
        theta_prime_masked_tokens = theta_forward_dist.sample()
        batch_idx = torch.arange(theta_prime_masked_tokens.shape[0]).unsqueeze(-1)
        if self.unique_in_batch: 
            theta_prime_tokens[batch_idx, masked_indices] = theta_prime_masked_tokens
            theta_masked_tokens = theta_tokens[batch_idx, masked_indices]
        else: 
            theta_prime_tokens[:, masked_indices] = theta_prime_masked_tokens
            theta_masked_tokens = theta_tokens[:, masked_indices]
        if self.mh:
            lp_forward = torch.sum(theta_forward_dist.log_prob(theta_prime_masked_tokens), dim=-1)
            theta_prime_onehot = nn.functional.one_hot(theta_prime_tokens, num_classes=self.vocab_size).float()
            grad_prime = self._calc_grad(model, theta_prime_onehot, attention_mask, masked_indices)
            t1_prime, t2_prime = self.calc_logits(theta_prime_tokens, grad_prime, masked_indices)
            theta_reverse_logits = balancing_constant * t1_prime - t2_prime / (2 * step_size)
            theta_reverse_dist = torch.distributions.Categorical(logits=theta_reverse_logits / self.proposal_temp)
            lp_reverse = torch.sum(theta_reverse_dist.log_prob(theta_masked_tokens), dim=-1)

            theta_prime_energy = model(theta_prime_onehot, attention_mask, masked_indices).squeeze()
            theta_energy = model(theta_onehot, attention_mask, masked_indices).squeeze()
            m_term = theta_prime_energy - theta_energy
            la = m_term + lp_reverse - lp_forward
            a = (la.exp() > torch.rand_like(la)).float()
            self.metrics['a_s'].append(a.mean().item())
            self.a_s.append(a.mean().item())
            self.metrics['proposed_energy'].append(theta_prime_energy.mean().item())
            self.metrics['orig_energy'].append(theta_energy.mean().item())
            theta_new_tokens = (theta_prime_tokens * a[:, None] + theta_tokens * (1 - a[:, None])).long()
        else:
            theta_new_tokens = theta_prime_tokens.long()
        theta['input_ids'] = theta_new_tokens
        theta['inputs'] = nn.functional.one_hot(theta_new_tokens.long(), num_classes=self.vocab_size).long()
        hops = ((theta_new_tokens != theta_tokens) * 1.).sum(dim=-1).mean().item()
        theta_tokens = theta_tokens.long()
        mse = torch.nn.functional.mse_loss(self.embeddings(theta_new_tokens), 
                                           self.embeddings(theta_tokens)).mean().item()
        self.metrics['mse_jump'].append(mse)
        self.metrics['hops'].append(hops)
        return theta

    def _step_no_cyclical(self, theta, model, **kwargs):
        return self.old_step(theta, model, 0)

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
        test_steps=1,
        lr=0.5,
        bal_resolution=3,
        dula_burnin=50,
        acs_burnin=0
    ):
        
        bdmala = AutomaticCyclicalSamplerOneHot(embeddings=self.embeddings, 
                                                **vars(self))
        bdmala.is_acs = False
        # a bit of manipulation so that the bdmala sampler being used for tuning 
        # does not require an index for the step 
        return_dict_key = "input_ids"
        bdmala.old_step = bdmala.step 
        bdmala.step = bdmala._step_no_cyclical
        bdmala.D = self.D
        bdmala.unique_in_batch = True
        bdmala.mh = False
        bdmala.step_size = 5
        bdmala.bal = init_big_bal
        bal_x_init = dula_x_init
        for i in range(dula_burnin):
           dula_x_init = bdmala.step(dula_x_init, model)
        bdmala.mh = True
        self.mh = True
        for i in range(acs_burnin): 
            dula_x_init = self.step(dula_x_init, model)
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
            return_dict_key=return_dict_key
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
            return_dict_key=return_dict_key
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
            init_small_bal = init_small_bal,
            return_dict_key = return_dict_key
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
    
    def _calc_grad(self, model, one_hot, attention_mask, masked_indices):
        
        one_hot = one_hot.float()
        one_hot=one_hot.detach()
        one_hot.requires_grad = True
        energy = model(inputs=one_hot, 
                       attention_mask=attention_mask, 
                       masked_indices=masked_indices).sum()
        grad = torch.autograd.grad(energy, one_hot)[0]
        return grad
    