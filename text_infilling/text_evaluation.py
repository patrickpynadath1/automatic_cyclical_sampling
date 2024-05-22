from transformers import GPT2LMHeadModel, GPT2TokenizerFast # for perplexity
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification # for cola 
import torch  
from nltk.translate.bleu_score import sentence_bleu
from nltk import word_tokenize
from typing import List
import numpy as np 


def load_cola_model(cola_model = "textattack/roberta-base-CoLA"):
    tokenizer = RobertaTokenizerFast.from_pretrained(cola_model)
    model = RobertaForSequenceClassification.from_pretrained(cola_model) 
    return tokenizer, model

def load_perplexity_model(gpt2_large="gpt2-large"):
    tokenizer = GPT2TokenizerFast.from_pretrained(gpt2_large)
    model= GPT2LMHeadModel.from_pretrained(gpt2_large)
    return tokenizer, model

def cola_generated(generated_sentences_text, cola_tokenizer, cola_model): 
    cola_pred = []
    cola_logits = []
    with torch.no_grad():
        for gen_sent_text in generated_sentences_text:
            for cur_sent in gen_sent_text: 
                inputs = cola_tokenizer(cur_sent, return_tensors="pt", padding=True)
                outputs = cola_model(**inputs)
                pred = outputs.logits.argmax(dim=1)
                logits = outputs.logits.softmax(dim=1)
                cola_logits.append(logits.cpu().numpy())
                cola_pred.append(pred.cpu().numpy())
    return cola_pred, cola_logits


def cola_baseline(generated_sentences_text, cola_tokenizer, cola_model): 
    cola_pred = []
    cola_logits = []
    with torch.no_grad():
        for cur_sent in generated_sentences_text:
            inputs = cola_tokenizer(cur_sent, return_tensors="pt", padding=True)
            outputs = cola_model(**inputs)
            pred = outputs.logits.argmax(dim=1)
            logits = outputs.logits.softmax(dim=1)
            cola_logits.append(logits.cpu().numpy())
            cola_pred.append(pred.cpu().numpy())
    return cola_pred, cola_logits


# the input for this function should be a list of generated sentences from 
# the same masked input 
def self_bleu_group(grouped_generated_sentences):
    total_self_bleu_scores = []
    for example in grouped_generated_sentences:
        for sentence_group in example: 
            group_scores = self_bleu(sentence_group)
            total_self_bleu_scores.append(group_scores)
    return total_self_bleu_scores

# expects tokens 
def perplexity(generated_sentences_text, tokenizer, model):
    perplexities = []
    for gen_sent_text in generated_sentences_text:
        cur_sentence_perplexities = []
        for cur_sent in gen_sent_text: 
            cur_sentence_perplexities.append(calculate_perplexity(cur_sent, model, tokenizer))
        perplexities.append(cur_sentence_perplexities)
    return perplexities


def calculate_perplexity(sentence, model, tokenizer):
    # Tokenize the input string
    input_ids = tokenizer.encode(sentence, return_tensors='pt')

    # Get the model's prediction
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)

    # Calculate the perplexity
    loss = outputs.loss
    perplexity = torch.exp(loss)

    return perplexity.item()


def self_bleu(generated_sentences: List[str]) -> List[float]:
    scores = []
    for i in range(len(generated_sentences)):
        reference = [word_tokenize(generated_sentences[j]) for j in range(len(generated_sentences)) if j != i] 
        candidate = word_tokenize(generated_sentences[i])
        score = sentence_bleu(reference, candidate)
        scores.append(score)
    return scores