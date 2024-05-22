import torch
from transformers import RobertaForMaskedLM, RobertaTokenizerFast
import torch.nn as nn

class TextInfillingEnergyFunction(nn.Module): 
    def __init__(self, model) -> None:
        super().__init__()
        self.model = model
        self.unique_in_batch = False

    def forward(self, tokens, attention_mask, masked_indices):
        output = self.model(input_ids = tokens, 
                            attention_mask = attention_mask)
        logits = output.logits
        masked_logits = logits[masked_indices]
        energies = masked_logits.log_softmax(dim=-1).sum(dim=-1) 
        return energies
    

class TextInfillingEnergyFunctionCE(TextInfillingEnergyFunction):
    def __init__(self, model="FacebookAI/roberta-base", input_type="embeds") -> None:
        super().__init__(model)
        self.input_type = input_type

    def forward(self, input, attention_mask, masked_indices): 
        CE = nn.CrossEntropyLoss()
        if self.input_type == "embeds":
            output = self.model(inputs_embeds = input, 
                                attention_mask = attention_mask)
        else:
            output = self.model(input_ids = input, 
                                attention_mask = attention_mask)
        logits = output.logits[:, masked_indices, :]
        return 
    

class TextInfillingEnergyFunctionOneHot(TextInfillingEnergyFunction): 
    
    def forward(self, 
                inputs, 
                attention_mask, 
                masked_indices):
        
        embeds = torch.einsum("bsv, ve -> bse", [inputs.float(), self.model.get_input_embeddings().weight])
        output = self.model(inputs_embeds = embeds,
                            attention_mask = attention_mask)
        logits = output.logits
        if self.unique_in_batch: 
            batched_index = torch.arange(masked_indices.shape[0]).unsqueeze(-1)
            masked_logits = logits[batched_index, masked_indices] * inputs[batched_index, masked_indices]
        else: 
            masked_logits = logits[:, masked_indices, :] * inputs[:, masked_indices, :]
        energies = torch.einsum('bsv -> b', [masked_logits])
        return energies 
    

class TextInfillingEnergyFunctionEmbeds(TextInfillingEnergyFunction): 
    
    def forward(self, 
                inputs, 
                attention_mask, 
                masked_indices):
        embed_map = self.model.get_input_embeddings().weight
        output = self.model(inputs_embeds = inputs,
                            attention_mask = attention_mask, 
                            output_hidden_states=True)
        if self.unique_in_batch: 
            batched_index = torch.arange(masked_indices.shape[0]).unsqueeze(-1)
            # shift by 1 to get the correct hidden states
            hidden_states = output.hidden_states[-1][batched_index, masked_indices]
            # computing the dot prod of hidden states and input embeds 
            dot_prod = torch.einsum("bse, bse -> bs", [hidden_states, inputs[batched_index, masked_indices]])
        else: 
            hidden_states = output.hidden_states[-1][:, masked_indices, :]
            dot_prod = torch.einsum("bse, bse -> bs", [hidden_states, inputs[:, masked_indices, :]])
        # computing the norm of the hidden states and input embeds
        vocab_dot_prod = torch.einsum("bse, ve -> bsv", [hidden_states, embed_map])
        # computes probs P(x_i | x_{j}, j != i) for masked tokens 
        energies = dot_prod - torch.logsumexp(vocab_dot_prod, dim=-1)
        return energies.sum(dim=-1) 