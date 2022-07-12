import random
import time
import os
import datetime
import numpy as np
import pandas as pd
from transformers import get_linear_schedule_with_warmup
from transformers import BertForSequenceClassification, AdamW, BertConfig
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data import TensorDataset, random_split
from transformers import AutoTokenizer
import torch
from torch import nn
from sentence_transformers import SentenceTransformer, util

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

pair_model_path = "model_save/hypo_pair_gold"
pair_model = BertForSequenceClassification.from_pretrained(pair_model_path)
tokenizer = AutoTokenizer.from_pretrained(pair_model_path)
pair_model = nn.DataParallel(pair_model)
pair_model.to(device)

judge_model_path = "model_save/hypo_judge_gold"
judge_model = BertForSequenceClassification.from_pretrained(judge_model_path)
judge_tokenizer = AutoTokenizer.from_pretrained(judge_model_path)
judge_model = nn.DataParallel(judge_model)
judge_model.to(device)


model = SentenceTransformer('model_save/hypo-pair-paraphrase-distilroberta-base-v1')


def get_hypo_pair_score(para, cands):
    paras = [para for i in range(len(cands))]
    sentences1 = paras
    sentences2 = cands
    #Compute embedding for both lists
    embeddings1 = model.encode(sentences1, convert_to_tensor=True)
    embeddings2 = model.encode(sentences2, convert_to_tensor=True)

    #Compute cosine-similarits
    cosine_scores = util.pytorch_cos_sim(embeddings1, embeddings2)
    #Output the pairs with their score
    res = [cosine_scores[i][i].item() for i in range(len(sentences1))]
    res = np.array(res)
    return res
    


"""
def get_hypo_pair_score(para, cands):
    # device = torch.device("cpu")
    paras = [para for i in range(len(cands))]
    # print(len(cands), end=',')
    encoded_inputs = tokenizer(
        cands, # the first sentence is hypo
        paras, # the second sentence is para
        add_special_tokens=True,
        max_length=256,
        truncation=True,
        padding='max_length',
        return_attention_mask=True,
        return_tensors='pt'
    )
    input_ids = encoded_inputs['input_ids'].to(device)
    attention_masks = encoded_inputs['attention_mask'].to(device)
    with torch.no_grad():
        result = pair_model(input_ids,
                       token_type_ids=None,
                       attention_mask=attention_masks,
                       return_dict=True)
    logits = result.logits.detach().cpu()
    m = nn.Softmax(dim=1)
    prob = m(logits)
    indices = torch.tensor([1])
    pos_prob = torch.index_select(prob, 1, indices)
    return pos_prob.flatten().numpy()
"""

def get_hypo_judge_score(cands):
    encoded_inputs = judge_tokenizer(
        cands,
        add_special_tokens=True,
        max_length=256,
        truncation=True,
        padding='max_length',
        return_attention_mask=True,
        return_tensors='pt'
    )
    input_ids = encoded_inputs['input_ids'].to(device)
    attention_masks = encoded_inputs['attention_mask'].to(device)
    with torch.no_grad():
        result = judge_model(input_ids,
                       token_type_ids=None,
                       attention_mask=attention_masks,
                       return_dict=True)
    logits = result.logits.detach().cpu()
    m = nn.Softmax(dim=1)
    prob = m(logits)
    indices = torch.tensor([1])
    pos_prob = torch.index_select(prob, 1, indices)
    #print("logits",logits)
    #print("prob", prob)
    #print("pos_prob", pos_prob)
    return pos_prob.flatten().numpy()





