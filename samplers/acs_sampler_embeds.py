from .acs_samplers import AutomaticCyclicalSampler
from .tuning_components import *
import torch 

class AutomaticCyclicalSamplerEmbeds(AutomaticCyclicalSampler):
    def __init__(self, 
                 **kwargs):
        super().__init__(**kwargs)
        self.embeddings = kwargs["embeddings"]
        self.proposal_temp = kwargs["proposal_temp"]
        self.is_acs = kwargs["is_acs"]
        # what mass is placed on the top proposed token? 
        self.metrics = {
            'top_mass': [], 
            'a_s': [],
            'proposed_energy': [],
            'orig_energy': [],
            'mse_jump':[],
            'hops': [] # for embeds, compute the mse between embeds 
        }
        self.a_s = []
        self.t1 = None 
        self.t2 = None 
        self.unique_in_batch = False


    def _calc_grad(self, model, embeds, attention_mask, masked_indices):
        embeds = embeds.detach()
        embeds.requires_grad = True 
        energy = model(embeds, attention_mask, masked_indices).sum()
        grad = torch.autograd.grad(energy, embeds)[0]
        return grad  
        
    # returns t1, t2 that get used to compute the DLP proposal distribution
    def _calc_logits(self, embeds, grad, indices):
        embed_map = self.embeddings.weight
        if self.unique_in_batch:  
            batch_idx = torch.arange(indices.shape[0]).unsqueeze(-1)
            embeds  = embeds[batch_idx, indices]
            grad = grad[batch_idx, indices]
        else: 
            embeds = embeds[:, indices]
            grad = grad[:, indices, :]
        # computing term1 = grad embeds * (potential_embeds - embeds)
        t1_1 = torch.einsum("bse, ev -> bsv", [grad, embed_map.T])
        t1_2 = torch.einsum("bse, bse -> bs", [embeds, grad])
        t1 = t1_1 - t1_2[:, :, None]
        # t1 = torch.einsum('abd, ab -> abd', [t1_1, -1 * t1_2])

        # computing ||theta' - theta|| using foil for conservation of gpu space
        t2_1 = torch.einsum("ve -> v", [embed_map**2]).repeat(
            embeds.size(0), embeds.size(1), 1
        )
        t2_2 = torch.einsum("bse, ev -> bsv", [embeds, embed_map.T])
        t2_3 = (
            torch.einsum("bsv -> bs", [embeds**2])
            .unsqueeze(-1)
            .repeat(1, 1, embed_map.size(0))
        )
        t2 = t2_1 - 2 * t2_2 + t2_3
        t2_scaled = (t2 - t2.min()) / t2.max()
        return t1, t2_scaled

    def step(self, theta, model, step_idx):
        if self.is_acs:
            step_size = self.step_sizes[step_idx % len(self.step_sizes)]
            balancing_parameter = self.balancing_constants[step_idx % len(self.balancing_constants)]
        else: 
            step_size = self.step_size
            balancing_parameter = .5
        EPS = 1e-10


        # unpacking what is needed 
        theta_embeds = theta['inputs']
        theta_tokens = theta['input_ids'].long()

        # regardless of the sampling step, these won't change 
        masked_ids = theta['masked_indices']
        attention_mask = theta['attention_mask']

        forward_grad = self._calc_grad(model, theta_embeds, attention_mask, masked_ids)
        t1_forward, t2_forward = self._calc_logits(
            theta_embeds, forward_grad, masked_ids
        )
        forward_delta = balancing_parameter * t1_forward - (1 / (2 * step_size))* (t2_forward)
        forward_dist = torch.distributions.Categorical(logits=(forward_delta)/self.proposal_temp )
        theta_prime_masked_tokens = forward_dist.sample()
        theta_prime_tokens = theta_tokens.clone()
        batch_idx = torch.arange(masked_ids.shape[0]).unsqueeze(-1)
        if self.unique_in_batch:
            theta_prime_tokens[batch_idx, masked_ids] = theta_prime_masked_tokens
            theta_masked_tokens = theta_tokens[batch_idx, masked_ids]
        else: 
            theta_prime_tokens[:, masked_ids] = theta_prime_masked_tokens
            theta_masked_tokens = theta_tokens[:, masked_ids]
        theta_prime_embeds = self.embeddings(theta_prime_tokens)
        # best_mass = (forward_delta/self.proposal_temp).softmax(dim=-1).detach().max(dim=-1).values.cpu().tolist()
        # self.metrics['top_mass'].append(best_mass)
        if self.mh:                 
            theta_prime_lp = torch.sum(forward_dist.log_prob(theta_prime_masked_tokens), dim=-1)

            theta_prime_grad = self._calc_grad(model, theta_prime_embeds, attention_mask, masked_ids)
            t1_reverse, t2_reverse = self._calc_logits(theta_prime_embeds, theta_prime_grad, masked_ids)
            reverse_delta = balancing_parameter * t1_reverse - (1 / (2 * step_size)) * (t2_reverse)
            reverse_dist = torch.distributions.Categorical(logits=(reverse_delta)/self.proposal_temp)
            theta_lp = torch.sum(reverse_dist.log_prob(theta_masked_tokens), dim=-1)
            
            theta_prime_energy =model(theta_prime_embeds, attention_mask, masked_ids).squeeze() 
            theta_energy = model(theta_embeds, attention_mask, masked_ids).squeeze() 
            m_term = theta_prime_energy - theta_energy
            la = m_term + theta_lp - theta_prime_lp
            a = (la.exp() > torch.rand_like(la)).float()
            theta_new_tokens = theta_prime_tokens * a[:, None] + theta_tokens * (1 - a[:, None])
            self.a_s.append(a.mean().item())
            self.metrics['a_s'].append(a.mean().item())
            self.metrics['proposed_energy'].append(theta_prime_energy.mean().item())
            self.metrics['orig_energy'].append(theta_energy.mean().item())
        else: 
            theta_new_tokens = theta_prime_tokens
        theta['input_ids'] = theta_new_tokens
        theta_new_embeds = self.embeddings(theta_new_tokens.long()).detach()
        theta['inputs'] = theta_new_embeds
        hops = ((theta_new_tokens != theta_tokens) * 1.).sum(dim=-1).mean().item()
        mse = torch.nn.functional.mse_loss(theta_new_embeds, theta_embeds).mean().item()
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
        
        bdmala = AutomaticCyclicalSamplerEmbeds(embeddings=self.embeddings, 
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
    

class AutomaticCyclicalSamplerEmbedsV2(AutomaticCyclicalSamplerEmbeds):
    def __init__(self, 
                 **kwargs):
        super().__init__(**kwargs)
        self.t1_forward_mean = []
        self.t2_forward_mean = []

        self.t1_backward_mean = []
        self.t2_backward_mean = []

    def compute_cycles(self):
        t1_forward_mean = torch.stack(self.t1_forward_mean).mean(dim=0)
        t2_forward_mean = torch.stack(self.t2_forward_mean).mean(dim=0)
        
        t1_backward_mean = torch.stack(self.t1_backward_mean).mean(dim=0)
        t2_backward_mean = torch.stack(self.t2_backward_mean).mean(dim=0)




    def step(self, theta, model, step_idx):
        if self.is_acs:
            step_size = self.step_sizes[step_idx % len(self.step_sizes)]
            balancing_parameter = self.balancing_constants[step_idx % len(self.balancing_constants)]
        else: 
            step_size = self.step_size
            balancing_parameter = .5
        EPS = 1e-10


        # unpacking what is needed 
        theta_embeds = theta['inputs']
        theta_tokens = theta['input_ids'].long()

        # regardless of the sampling step, these won't change 
        masked_ids = theta['masked_indices']
        attention_mask = theta['attention_mask']

        forward_grad = self._calc_grad(model, theta_embeds, attention_mask, masked_ids)
        t1_forward, t2_forward = self._calc_logits(
            theta_embeds, forward_grad, masked_ids
        )
        forward_delta = balancing_parameter * t1_forward - (1 / (2 * step_size))* (t2_forward)
        forward_dist = torch.distributions.Categorical(logits=(forward_delta)/self.proposal_temp )
        theta_prime_masked_tokens = forward_dist.sample()
        theta_prime_tokens = theta_tokens.clone()
        batch_idx = torch.arange(masked_ids.shape[0]).unsqueeze(-1)
        if self.unique_in_batch:
            theta_prime_tokens[batch_idx, masked_ids] = theta_prime_masked_tokens
            theta_masked_tokens = theta_tokens[batch_idx, masked_ids]
        else: 
            theta_prime_tokens[:, masked_ids] = theta_prime_masked_tokens
            theta_masked_tokens = theta_tokens[:, masked_ids]
        theta_prime_embeds = self.embeddings(theta_prime_tokens)
        # best_mass = (forward_delta/self.proposal_temp).softmax(dim=-1).detach().max(dim=-1).values.cpu().tolist()
        # self.metrics['top_mass'].append(best_mass)
        if self.mh:                 
            theta_prime_lp = torch.sum(forward_dist.log_prob(theta_prime_masked_tokens), dim=-1)

            theta_prime_grad = self._calc_grad(model, theta_prime_embeds, attention_mask, masked_ids)
            t1_reverse, t2_reverse = self._calc_logits(theta_prime_embeds, theta_prime_grad, masked_ids)
            reverse_delta = balancing_parameter * t1_reverse - (1 / (2 * step_size)) * (t2_reverse)
            reverse_dist = torch
    

    