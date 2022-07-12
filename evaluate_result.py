import json
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime

import nltk  # Here to have a nice missing dependency error message early on
import numpy as np
import pandas as pd
from datasets import load_dataset, load_metric
from unexpected_mask import unexpected_sent_score, get_diff_score

from hypo_select import get_hypo_pair_score, get_hypo_judge_score
from filelock import FileLock
from bert_score import BERTScorer
from nltk.translate.bleu_score import corpus_bleu
from nltk import word_tokenize
#from hypo_select import get_hypo_pair_score, get_hypo_judge_score

def get_length(s):
    return len(word_tokenize(s))

def compute_metrics(baseline, test_paras, cands):
    assert len(test_paras) == len(cands)
    cands = [[c] for c in cands]

    para_cand_map = dict(zip(test_paras, cands))



    df = pd.read_csv("HYPO.tsv", delimiter='\t',
                     header=None, names=['hypo', 'para', 'mnc'])
    df = df.dropna(subset=['hypo', 'para'])
    para = df.para.values.tolist()
    para = [s.strip() for s in para]
    refs = df.hypo.values.tolist()
    refs = [s.strip() for s in refs]
    # assert len(refs) == len(list(para_cand_map.keys()))

    para_ref_map = dict(zip(para,refs))
    para_hypo = []

    scorer = BERTScorer(lang="en", rescale_with_baseline=True, device='cpu')
    df_cand_scores = pd.DataFrame(columns=['para','cand', 'hypo_score', 'pair_score', 'unexpected_score', 'select_score'])
    refs = []
    gens = []
    best_cand_scores = []
    repetition_idx = []
    for idx, para in enumerate(para_cand_map.keys()):
        cands = para_cand_map[para]
        unexpected_scores = np.array([unexpected_sent_score(s) for s in cands])
        pair_scores = get_hypo_pair_score(para, cands)
        hypo_scores = get_hypo_judge_score(cands)
        select_scores = pair_scores + hypo_scores
        # select_scores = pair_scores + hypo_scores + unexpected_scores
        # select_scores = (hypo_scores>0.5)*pair_scores*hypo_scores
        # select_scores = pair_scores
        cand_scores = [] 
        for i in range(len(cands)):
            cand_scores.append([para, cands[i], hypo_scores[i], pair_scores[i], unexpected_scores[i], select_scores[i]])
        df_tmp = pd.DataFrame(cand_scores, columns=['para','cand', 'hypo_score', 'pair_score', 'unexpected_score','select_score'])
        df_tmp = df_tmp.sort_values(by=['select_score', 'hypo_score', 'unexpected_score','pair_score'],  ascending=[False, False, False, False])
        df_cand_scores = df_cand_scores.append(df_tmp)
        
        if not select_scores.any(): # all zeros
            max_idx = np.argmax(hypo_scores)
            best_cand = cands[max_idx]
        else:
        
            max_idx = np.argmax(select_scores)
            best_cand = cands[max_idx]
        """
        if idx in [2, 4, 7, 9, 10, 11, 12, 15, 16, 17, 20, 22, 24, 25, 27, 28, 29, 30, 32, 34, 37, 38, 41, 45, 52, 53, 54, 55, 57, 58, 60, 62, 64, 66, 67, 69]:
            # print(para, best_cand)
            continue
        """
        if best_cand == para:
            repetition_idx.append(idx)
            # continue
        diff_score = get_diff_score(para, best_cand)
        length = get_length(best_cand)
        best_cand_scores.append([hypo_scores[max_idx], unexpected_scores[max_idx], pair_scores[max_idx], select_scores[max_idx], diff_score, length])

        para_hypo.append([para,best_cand])
        gens.append(best_cand)
        refs.append(para_ref_map[para])
    print("copy:", len(repetition_idx), "total:", len(refs))
    P, R, F1 = scorer.score(gens, refs)
    bert_score = F1.tolist()

    metrics = dict()
    #max_test_samples = data_args.max_test_samples if data_args.max_test_samples is not None else len(test_dataset)
    #metrics["test_samples"] = min(max_test_samples, len(test_dataset))
    metrics["BERTScore"] = sum(bert_score)/len(bert_score)
    r = [[row.split()] for row in refs]
    c = [row.split() for row in gens]
    metrics["BLEU"] = corpus_bleu(r, c)*100
    metrics["BLEU-1"] = corpus_bleu(r, c,weights=(1, 0, 0, 0))*100
    metrics["BLEU-2"] = corpus_bleu(r, c,weights=(0, 1, 0, 0))*100 
    metrics["BLEU-3"] = corpus_bleu(r, c,weights=(0, 0, 1, 0))*100
    metrics["BLEU-4"] = corpus_bleu(r, c,weights=(0, 0, 0, 1))*100
    best_cand_scores = np.array(best_cand_scores)
    best_cand_scores_avg = np.average(best_cand_scores, axis=0)
    metrics['hypo_score'] = best_cand_scores_avg[0]
    metrics['unexpected_score'] = best_cand_scores_avg[1]
    metrics['pair_score'] = best_cand_scores_avg[2]
    metrics['select_score'] = best_cand_scores_avg[3]
    metrics["difference"] =  best_cand_scores_avg[4]
    metrics["length"] =  best_cand_scores_avg[5]
    # metrics["EM"] = sum([gen == para for gen, para in zip(gens, test_paras)]) / len(gens)
    metrics["EM"] = sum([item[0] == item[1] for item in para_hypo]) / len(gens)
    print("*"*20, baseline, "*"*20)
    print(metrics)

df = pd.read_csv("bart_hypo_hypo_test.csv", header=None, names=['para', 'cand'])
test_paras = df.para.values.tolist()
#test_cands = df.cand.values.tolist()
#compute_metrics("BART-HYPO", test_paras, test_cands)


output_file = "HYPO_test_for_eval.csv"
df = pd.read_csv(output_file)
mover = df.MOVER.values.tolist()
R1 = df.R1.values.tolist()
R3 = df.R3.values.tolist()
BART = df.BART.values.tolist()
BART_PARA = df.BART_PARA.values.tolist()
HUMAN = df.HUMAN.values.tolist()
NAIVE = test_paras

df = pd.read_csv("model_save/bart_base_/checkpoint-10479/test_final_output.csv")
MOVER = df['hyperbolic sentence'].tolist()


df = pd.read_csv("model_save/bart_hypo_hypo_test/test_final_output.csv", header=None, names=['literal sentence', 'hyperbolic sentence'])
BART_HYPO = df['hyperbolic sentence'].tolist()

df = pd.read_csv("model_save/mover_test_no_tag/test_final_output.csv")
MOVER_HYPO_SELECT = df['hyperbolic sentence'].tolist()

df = pd.read_csv("model_save/mover_hypo_train/test_final_output.csv")
MOVER_HYPO_TRAIN = df['hyperbolic sentence'].tolist()

df = pd.read_csv("model_save/mover_hypo_train_hypo-xl/test_final_output.csv")
MOVER_HYPO_TRAIN_HYPO_XL = df['hyperbolic sentence'].tolist()




"""
with open("model_save/debug_bart_hypo_whole_input/test_generations.txt", 'r') as inF:
    BART_HYPO_REAL = [l.strip() for l in inF.readlines()]
assert NAIVE != BART_HYPO_REAL # asserted!
"""
with open("model_save/mover_bart_hypo/test_generations.txt", 'r') as inF:
    MOVER_BART_HYPO = [l.strip() for l in inF.readlines()]
assert NAIVE != MOVER_BART_HYPO
with open("output/paraphrase_generations.txt", 'r') as inF:
    PARA = [l.strip() for l in inF.readlines()]
assert NAIVE != PARA
with open("model_save/debug_bart_base_para_paws_qqp_hypo/test_generations.txt", 'r') as inF:
    BART_PARA_HYPO = [l.strip() for l in inF.readlines()]
assert NAIVE != BART_PARA_HYPO
with open("model_save/debug_bart_base_para_paws_qqp_whole_input/test_generations.txt", 'r') as inF:
    BART_PARA_PAWS_QQP = [l.strip() for l in inF.readlines()]
assert NAIVE != BART_PARA_PAWS_QQP



#compute_metrics("BART_HYPO_REAL", test_paras, BART_HYPO_REAL)
compute_metrics("MOVER", test_paras, MOVER)
compute_metrics("R1", test_paras, R1)
compute_metrics("R3", test_paras, R3)
compute_metrics("BART", test_paras, BART)
#compute_metrics("BART-PARA", test_paras, BART_PARA)
compute_metrics("HUMAN", test_paras, HUMAN)
compute_metrics("NAIVE", test_paras, NAIVE)
#compute_metrics("BART_HYPO", test_paras, BART_HYPO)
#compute_metrics("MOVER_BART_HYPO", test_paras, MOVER_BART_HYPO)
#compute_metrics("MOVER_HYPO_SELECT", test_paras, MOVER_HYPO_SELECT)
#compute_metrics("MOVER_HYPO_TRAIN", test_paras, MOVER_HYPO_TRAIN)
#compute_metrics("MOVER_HYPO_TRAIN_HYPO_XL", test_paras, MOVER_HYPO_TRAIN_HYPO_XL)
#compute_metrics("PARA", test_paras, PARA)
#compute_metrics("BART_PARA_HYPO", test_paras, BART_PARA_HYPO)
#compute_metrics("BART_PARA_PAWS_QQP", test_paras, BART_PARA_PAWS_QQP)
# test_paras = df.para.values.tolist()

"""
fIn=open("bart_para_outputs.txt", 'r')
cands =[[l.strip()] for l in fIn.readlines()]
"""

# cands = [[p.strip()] for p in test_paras]

# cands = df.cand.values.tolist()


