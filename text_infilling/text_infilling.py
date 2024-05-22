import torch
from transformers import RobertaForMaskedLM, RobertaTokenizerFast
import torch.nn as nn
from datasets import load_dataset 
import pandas as pd
import spacy
import random 


def load_base_model(): 
    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base", add_prefix_space=True)
    model = RobertaForMaskedLM.from_pretrained("roberta-base")
    return tokenizer, model  


# TODO: 
# make sure special characters are removed and not tokenized 
# make sure that the lengths are accurate
# make sure that the mask indices are within the specific range of 
# the actual text  

class InfillingDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 text_data, 
                 tokenizer, 
                 embed_table = None,
                 max_length=512, 
                 mask_probability=0.5,
                 number_masks=None):
        self.text_data = text_data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mask_probability = mask_probability
        self.number_masks = number_masks
        self.embed_table = embed_table

    def __len__(self):
        return len(self.text_data)
    
    def __getitem__(self, idx):
        tokenizer_out = self.tokenizer(self.text_data[idx], return_tensors='pt') 
        tokens = tokenizer_out["input_ids"]
        attention_mask = tokenizer_out["attention_mask"]
        # adding the mask token to specificed indices 
        mask_indices = self._generate_random_mask_indices(tokens)
        tokens[0, mask_indices] = self.tokenizer.mask_token_id
        if self.embed_table is not None: 
            inputs = self.embed_table(tokens.cuda()).detach()
        else: 
            inputs = torch.nn.functional.one_hot(tokens, num_classes=self.tokenizer.vocab_size).float()
        return {
            'input_ids': tokens,
            'inputs': inputs,
            'attention_mask': attention_mask,
            'masked_indices': mask_indices,
        }
    
    def _generate_random_mask_indices(self, tokens):
        length = tokens.size(-1)
        if self.number_masks is not None: 
            number_masks = self.number_masks
        else: 
            number_masks = int(length*self.mask_probability)
        random_indices = list(random.sample(range(1, length-1), number_masks))
        random_indices.sort()
        #random_indices = torch.randint(1, length-1, (int(length*self.mask_probability),)).sort().values
        return torch.Tensor(random_indices).long()
    

# returns list of sentences 
def load_text_data(dataset_name):
    if dataset_name == "grimm":  
        ds = load_grimm_dataset()
    elif dataset_name == "sst2": 
        ds = load_sst2_dataset()
    elif dataset_name == "rte": 
        ds = load_rte_dataset()
    elif dataset_name == "wnli": 
        ds = load_wnli_dataset()
    elif dataset_name == "qnli": 
        ds = load_qnli_dataset()
    return ds


# raw text dataset = list of sentences (List[Str])
def load_grimm_dataset():
    df = pd.read_csv("text_data/grimms_fairytales.csv")
    parsed_words = []
    nlp = spacy.load('en_core_web_sm')
    for i, row in df.iterrows(): 
        text = row["Text"]
        story = row["Title"]
        parsed = nlp(text)
        for cur_sent in parsed.sents: 
            parsed_words.append(list([tok.text for tok in cur_sent]))
    cleaned_sents = []
    for sent in parsed_words:
        temp_sent = []
        for w in sent: 
            if w == "\n": 
                continue
            temp_sent.append(w)
        full_sent = " ".join(temp_sent)
        cleaned_sents.append(full_sent) 
    return cleaned_sents


def load_sst2_dataset(): 
    df = pd.read_csv("text_data/SST-2/dev.tsv", sep="\t", header=0)
    parsed_words = []
    nlp = spacy.load('en_core_web_sm')
    for i, row in df.iterrows(): 
        text = row["sentence"]
        cur_sent = nlp(text)
        parsed_words.append(list([tok.text for tok in cur_sent]))
    cleaned_sents = []
    for sent in parsed_words:
        temp_sent = []
        for w in sent: 
            if w == "\n": 
                continue
            temp_sent.append(w)
        full_sent = " ".join(temp_sent)
        cleaned_sents.append(full_sent)
    return cleaned_sents



def load_rte_dataset(): 
    df = pd.read_csv("text_data/RTE/dev.tsv", sep="\t", header=0)
    parsed_words = []
    nlp = spacy.load('en_core_web_sm')
    for i, row in df.iterrows(): 
        text = row["sentence1"]
        cur_sent = nlp(text)
        parsed_words.append(list([tok.text for tok in cur_sent]))
    cleaned_sents = []
    for sent in parsed_words:
        temp_sent = []
        for w in sent: 
            if w == "\n": 
                continue
            temp_sent.append(w)
        full_sent = " ".join(temp_sent)
        cleaned_sents.append(full_sent)
    return cleaned_sents


def load_wnli_dataset(): 
    df = pd.read_csv("text_data/WNLI/dev.tsv", sep="\t", header=0)
    parsed_words = []
    nlp = spacy.load('en_core_web_sm')
    for i, row in df.iterrows(): 
        for j in range(2): 
            text = row[f"sentence{j+1}"]
            cur_sent = nlp(text)
            parsed_words.append(list([tok.text for tok in cur_sent]))
    cleaned_sents = []
    for sent in parsed_words:
        temp_sent = []
        for w in sent: 
            if w == "\n": 
                continue
            temp_sent.append(w)
        full_sent = " ".join(temp_sent)
        cleaned_sents.append(full_sent)
    return cleaned_sents


def load_qnli_dataset(): 
    df = pd.read_csv("text_data/QNLI/dev.tsv", sep="\t", header=0)
    parsed_words = []
    nlp = spacy.load('en_core_web_sm')
    for i, row in df.iterrows(): 
        text = row["sentence"]
        cur_sent = nlp(text)
        parsed_words.append(list([tok.text for tok in cur_sent]))
    cleaned_sents = []
    for sent in parsed_words:
        temp_sent = []
        for w in sent: 
            if w == "\n": 
                continue
            temp_sent.append(w)
        full_sent = " ".join(temp_sent)
        cleaned_sents.append(full_sent)
    return cleaned_sents