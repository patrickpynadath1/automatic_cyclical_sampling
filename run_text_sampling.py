from text_infilling import *
from samplers import AutomaticCyclicalSamplerEmbeds, AutomaticCyclicalSamplerOneHot
from config_cmdline import config_acs_args, config_acs_pcd_args
from argparse import ArgumentParser
import pickle
import os
import yaml
import tqdm


def generate_text_samples(
    tokenizer,
    text_dataset,
    num_samples,
    dmala,
    acs,
    energy_function,
    num_steps,
    subsample,
    batch_size
):
    # how to handle the burnin for ACS?
    total_metrics = {}
    total_metrics['dmala'] = []
    total_metrics['acs'] = []
    generated_samples = {}
    generated_samples['dmala'] = [[] for _ in range(len(text_dataset))]
    generated_samples['acs'] = [[] for _ in range(len(text_dataset))]
    generated_samples['masked_indices'] = []
    output_energies_total = []
    for data_idx in range(len(text_dataset)):
        try: 
            input_dct = text_dataset[data_idx]
            generated_samples['masked_indices'].append(input_dct['masked_indices'])
            outputs, output_energies = generate_sample(
                input_dct = input_dct,
                dmala = dmala,
                acs = acs,
                energy_function=energy_function,
                num_steps = num_steps,
                subsample=subsample,
                batch_size=batch_size
            )
            output_energies_total.append(output_energies)
            total_metrics["dmala"].append(dmala.metrics)
            total_metrics['acs'].append(acs.metrics)
            for j, sampler_name in enumerate(['dmala', 'acs']):
                print(f"Sampler: {sampler_name}")
                for i in range(outputs['acs'].shape[0]):
                    final_text = tokenizer.decode(outputs[sampler_name][i].long())
                    generated_samples[sampler_name][data_idx].append(final_text)
                    print(final_text)
        except Exception as e:
            print(f"Error: {e}")
    # run sampling loop, collect generated samples

    return generated_samples, total_metrics, output_energies_total


def generate_sample(
    input_dct,
    dmala,
    acs,
    energy_function,
    num_steps,
    batch_size,
    subsample
):  
    
    # TODO: make a copy of this so thatk 
    input_dct["inputs"] = (
        input_dct["inputs"].repeat(batch_size, 1, 1).detach()
    )
    input_dct["input_ids"] = (
        input_dct["input_ids"].repeat(batch_size, 1)
    )
    input_dct["attention_mask"] = (
        input_dct["attention_mask"].repeat(batch_size, 1)
    )
    acs_inputs = {}
    acs_inputs['inputs'] = input_dct['inputs'].clone()
    acs_inputs['input_ids'] = input_dct['input_ids'].clone()
    acs_inputs['attention_mask'] = input_dct['attention_mask'].clone()
    acs_inputs['masked_indices'] = input_dct['masked_indices'].clone()
    samplers = [acs, dmala]
    sampler_names = ['acs', 'dmala']
    sampler_inputs = [acs_inputs, input_dct]
    final_outputs = {}
    final_energies_total = {}
    for i in range(2): 
        inputs = sampler_inputs[i]
        for k in ['inputs', 'attention_mask', 'input_ids', 'masked_indices']: 
            inputs[k] = inputs[k].cuda()
        sampler = samplers[i]
        cur_samples = []
        for step_idx in tqdm.tqdm(range(num_steps)):
            inputs = sampler.step(inputs, energy_function, step_idx)
            if subsample != -1 and step_idx % subsample == 0:
                cur_samples.append(inputs['input_ids'].detach().cpu())
        cur_samples.append(inputs['input_ids'].detach().cpu())
        final_energies = energy_function(inputs['inputs'], 
                                         inputs['attention_mask'],
                                         inputs['masked_indices']).detach().cpu().tolist()
        # deallocating the memory on the gpu 
        for k in ["inputs", "input_ids", "attention_mask"]: 
            inputs[k] = inputs[k].cpu()
        final_outputs[sampler_names[i]] = torch.cat(cur_samples, dim=0)
        final_energies_total[sampler_names[i]] = final_energies
    return final_outputs, final_energies_total


def run_evaluation_loop(
    original_sentences, generated_sentences, num_examples, num_samples
):

    # load the models
    cola_tokenizer, cola_model = load_cola_model()
    gpt2_tokenizer, gpt2_model = load_perplexity_model()

    # baselines for cola, perplexities
    cola_pred_baseline, cola_logits_baseline = cola_baseline(
        original_sentences, cola_tokenizer, cola_model
    )
    perplexities_baseline = []
    for orig_sent in original_sentences: 
        perplexities_baseline.append(calculate_perplexity(orig_sent, gpt2_model, gpt2_tokenizer))

    # examples of interest
    cola_pred, cola_logits = cola_generated(generated_sentences, cola_tokenizer, cola_model)
    perplexities = perplexity(generated_sentences, gpt2_tokenizer, gpt2_model)
    grouped_examples = []
    for i in range(num_examples):
        grouped_examples.append(
            generated_sentences[i * num_samples : (i + 1) * num_samples]
        )
    self_bleu_scores = self_bleu_group(grouped_examples)

    data_res = {
        "baseline_data": {
            "cola_pred": cola_pred_baseline,
            "cola_logits": cola_logits_baseline,
            "perplexities": perplexities_baseline,
        },
        "generated_data": {
            "cola_pred": cola_pred,
            "cola_logits": cola_logits,
            "perplexities": perplexities,
            "self_bleu": self_bleu_scores,
        },
    }
    for key, value in data_res["baseline_data"].items():
        print(f"Baseline {key}: {value}")
    for key, value in data_res["generated_data"].items():
        print(f"Generated {key}: {value}")

    # with open(f"{save_dir}/evaluation_res.pickle", "wb") as f:
    #     pickle.dump(data_res, f)

    return data_res


def get_dlp_samplers(temp, dim, device, embeddings, args):
    use_mh = temp == "dmala" or temp == "acs"
    is_acs = False
    if temp == "acs":
        is_acs = True
    if args.use_embeds:
        ACS_sampler_class = AutomaticCyclicalSamplerEmbeds
    else:
        ACS_sampler_class = AutomaticCyclicalSamplerOneHot
    sampler = ACS_sampler_class(
        dim=dim,
        n_steps=args.n_steps,
        num_cycles=args.num_cycles,
        initial_balancing_constant=1.0,
        fixed_proposal=False,
        approx=True,
        multi_hop=False,
        temp=1.0,
        mean_stepsize=args.step_size,
        mh=use_mh,
        num_iters=args.n_steps,
        device=device,
        burnin_adaptive=True,
        embeddings=embeddings,
        proposal_temp=args.proposal_temp,
        is_acs=is_acs,
        vocab_size=embeddings.weight.shape[0],
    )

    return sampler


def configure_save_dir(args):
    running_dir_name = "text_infill"
    # running_dir_name += args.sampler
    if args.use_embeds:
        running_dir_name += "_embeds"
    save_dir = f"{args.save_dir}_{args.dataset_name}/{running_dir_name}"
    if args.run_name != "":
        save_dir += f"_{args.run_name}"
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
    else:
        counter = 1
        while os.path.exists(f"{save_dir}_{counter}"):
            counter += 1
        save_dir = f"{save_dir}_{counter}"
        os.mkdir(save_dir)
        os.mkdir(f"{save_dir}/dmala")
        os.mkdir(f"{save_dir}/acs")
    return save_dir


def dump_arg_configs(args, save_dir):
    arg_dct = vars(args)
    with open("config.yaml", "w") as f:
        yaml.dump(arg_dct, f"{save_dir}/config.yaml")
    return


def config_burnin_set(text_data, 
                      tokenizer,
                      num_burnin_samples=128,
                      num_generative_examples=100,
                      min_length = 15, 
                      max_length=50):
    potential_indices = []
    lengths = []
    for idx in range(len(text_data)):
        out = tokenizer(text_data[idx], return_tensors='pt') 
        sent_length = out['input_ids'].size(-1)
        if sent_length > min_length and sent_length < max_length: 
            potential_indices.append(idx)
            lengths.append(sent_length)
    burnin_indices = random.sample(potential_indices, num_burnin_samples)
    print(f"burnin potenital ex: {len(potential_indices)}")
    new_potential_indices = []
    for i in potential_indices: 
        if i not in burnin_indices: 
            new_potential_indices.append(i)
    print(f"potential generative ex: {len(new_potential_indices)}")
    generative_indices = random.sample(new_potential_indices, num_generative_examples)
    return burnin_indices, generative_indices

# no returns, just edits the parameters of the
# # ACS sampler, 
def run_acs_burnin(text_data, 
                   tokenizer,
                   energy_function, 
                   sampler,
                   budget,
                   burnin_data_indices,
                   embed_table,
                   use_embeds,
                   max_length, 
                   number_masks):
    # how to handle the burnin for ACS?
    # idea: need to have multiple different sentences 
    # so that the parameters estimated generalize well
    vocab_size = embed_table.weight.shape[0]
    embed_size = embed_table.weight.shape[1]
    if use_embeds:
        burnin_dataset = InfillingDataset(text_data, 
                                        tokenizer=tokenizer,
                                        embed_table=embed_table,
                                        number_masks=number_masks)
    else: 
        burnin_dataset = InfillingDataset(text_data, 
                                        tokenizer=tokenizer,
                                        number_masks=number_masks)
    if use_embeds: 
        burnin_theta = torch.zeros(len(burnin_data_indices), max_length, embed_size)
    else:
        burnin_theta = torch.zeros(len(burnin_data_indices), max_length, vocab_size)
    masked_indices = torch.zeros(len(burnin_data_indices), number_masks)
    tokens = torch.zeros(len(burnin_data_indices), max_length)
    attention_mask = torch.zeros_like(tokens)
    # configuring the masked indices 
    for target_idx, data_idx in enumerate(burnin_data_indices):
        input_dct = burnin_dataset[data_idx]
        sentence_length = input_dct["input_ids"].shape[1]
        burnin_theta[target_idx, :sentence_length, :] = input_dct["inputs"]
        tokens[target_idx, :sentence_length] = input_dct["input_ids"]
        masked_indices[target_idx, :] = input_dct["masked_indices"]
        attention_mask[target_idx, :sentence_length] = input_dct["attention_mask"]
    burnin_theta = burnin_theta.cuda()
    masked_indices = masked_indices.cuda()
    
    # running the burnin
    burnin_dct = {
        'inputs': burnin_theta.cuda(),
        'attention_mask': attention_mask.cuda(), 
        'masked_indices': masked_indices.long().cuda(),
        'input_ids': tokens.long().cuda()
    }
    energy_function.unique_in_batch = True
    _, burnin_res = sampler.tuning_alg(burnin_dct,
                       energy_function, 
                       budget, 
                       init_big_step = 5, 
                       init_small_step = .1, 
                       init_big_bal = .85)

    energy_function.unique_in_batch = False
    for k, v in burnin_dct.items(): 
        burnin_dct[k] = v.cpu()
    return burnin_res


def main(args):
    save_dir = configure_save_dir(args)
    # dump_arg_configs(args, save_dir)
    cur_seed = args.seed
    # fixing the seed for reproducibility 
    torch.manual_seed(cur_seed)
    np.random.seed(cur_seed)
    random.seed(cur_seed)
    complete_dataset = load_text_data(args.dataset_name)
    tokenizer, model = load_base_model()
    model = model.cuda()
    burnin_indices, potential_indices = config_burnin_set(text_data=complete_dataset,
                                       tokenizer=tokenizer, 
                                       num_burnin_samples=args.burnin_samples,
                                       num_generative_examples=args.examples,
                                       min_length=args.burnin_min_length, 
                                       max_length=args.burnin_max_length)
    energy_function = TextInfillingEnergyFunctionEmbeds(model) if args.use_embeds else TextInfillingEnergyFunctionOneHot(model)
    dmala_sampler = get_dlp_samplers(
        'dmala', tokenizer.vocab_size, "cuda:0", model.get_input_embeddings(), args
    )
    acs_sampler = get_dlp_samplers(
        'acs', tokenizer.vocab_size, "cuda:0", model.get_input_embeddings(), args
    )
    acs_burnin_res = run_acs_burnin(text_data=complete_dataset,
                    tokenizer=tokenizer,
                    energy_function=energy_function,
                    sampler=acs_sampler,
                    budget=args.burnin_budget,
                    burnin_data_indices=burnin_indices,
                    embed_table=model.get_input_embeddings(),
                    use_embeds=args.use_embeds,
                    max_length=args.burnin_max_length,
                    number_masks=args.burnin_number_masks)
    pickle.dump(acs_burnin_res, open(f"{save_dir}/acs/burnin_res.pickle", "wb"))

    
    text_data = [] 
    for i in range(len(complete_dataset)):
        if i in potential_indices: 
            text_data.append(complete_dataset[i])
    if args.use_embeds:
        text_dataset = InfillingDataset(
            text_data, tokenizer, embed_table=model.get_input_embeddings(), 
            mask_probability=args.inference_mask_p
        )
    else:
        text_dataset = InfillingDataset(text_data, tokenizer, mask_probability = args.inference_mask_p)
    generated_samples, total_metrics, final_energies = generate_text_samples(
        tokenizer=tokenizer,
        text_dataset=text_dataset,
        num_samples=args.samples_per_example,
        dmala=dmala_sampler,
        acs=acs_sampler,
        energy_function=energy_function,
        num_steps=args.n_steps, 
        subsample=args.subsample, 
        batch_size=args.batch_size
    )
    data_res_dmala = run_evaluation_loop(
        text_data,
        generated_samples['dmala'],
        num_samples=args.samples_per_example,
        num_examples=args.examples,
    )
    data_res_acs = run_evaluation_loop(
        text_data,
        generated_samples['acs'],
        num_samples=args.samples_per_example,
        num_examples=args.examples,
    )
    pickle.dump(generated_samples['masked_indices'], open(f"{save_dir}/masked_indices.pickle", "wb"))
    pickle.dump(data_res_acs, open(f"{save_dir}/acs/evaluation_res.pickle", "wb"))
    pickle.dump(total_metrics['acs'], open(f"{save_dir}/acs/total_metrics.pickle", "wb"))
    pickle.dump(data_res_dmala, open(f"{save_dir}/dmala/evaluation_res.pickle", "wb"))
    pickle.dump(total_metrics['dmala'], open(f"{save_dir}/dmala/total_metrics.pickle", "wb"))
    pickle.dump(final_energies, open(f"{save_dir}/final_energies.pickle", "wb"))
    # pickle.dump(final_energies['acs'], open(f"{save_dir}/dmala/final_energies.pickle", "wb"))
    with open(f"{save_dir}/dmala/generated_text.txt", "w") as f:
        for ex in generated_samples['dmala']: 
            for text in ex:
                f.write(text + "\n")

    with open(f"{save_dir}/acs/generated_text.txt", "w") as f:
        for ex in generated_samples['acs']: 
            for text in ex:
                f.write(text + "\n")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser = config_acs_args(parser)
    parser.add_argument("--burnin_samples", type=int, default=32)
    parser.add_argument("--burnin_min_length", type=int, default=20)
    parser.add_argument("--burnin_number_masks", type=int, default=4)
    parser.add_argument("--burnin_max_length", type=int, default=80)
    parser.add_argument("--inference_mask_p", type=float, default=.35)
    parser.add_argument("--seed", type=int, default=1234567)
    parser.add_argument("--examples", type=int, default=200)
    parser.add_argument("--subsample", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=25)
    parser.add_argument("--samples_per_example", type=int, default=10)
    parser.add_argument("--save_dir", type=str, default="raw_exp_data/text_infilling")
    parser.add_argument("--sampler", type=str, default="acs")
    parser.add_argument("--step_size", type=float, default=.5)
    parser.add_argument("--n_steps", type=int, default=25)
    parser.add_argument("--proposal_temp", type=float, default=1.0)
    parser.add_argument("--use_embeds", action="store_true")
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--dataset_name", type=str, default='sst2')
    parser.add_argument("--num_examples", type=int, default=1)

    args = parser.parse_args()
    main(args)
