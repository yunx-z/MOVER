import os
import pandas as pd
from tqdm import tqdm
from semantic_search import get_hypo_cands
from unexpected_syntax_mask import get_masked_spans_syntax, search_for_syntax
from hypo_select import get_hypo_pair_score, get_hypo_judge_score
import numpy as np

def replace_hypo(para, cands):
    #print(para)
    #print(cands)

    hypos = []
    for cand in cands:
        masked_spans_syntax_list = get_masked_spans_syntax(cand)
        for hypo_span, syntax in masked_spans_syntax_list:
            results = search_for_syntax(para, syntax)
            for result in results:
                left, match, right = result
                hypo = ' '.join([left, hypo_span, right])
                hypos.append(hypo)

    
    #print(hypos)
    best_cand = None
    if len(hypos)==0:
        # fall back to retrieve
        best_cand =  cands[0]
    else:
        pair_scores = get_hypo_pair_score(para, hypos)
        hypo_scores = get_hypo_judge_score(hypos)
        select_scores = hypo_scores * ((1.0 - pair_scores) > 1e-3) * (pair_scores > 0.8) # 生成原句则为0
        if not select_scores.any(): # all zeros
            best_cand = cands[np.argmax(hypo_scores)]
        else:
            max_idx = np.argmax(select_scores)
            best_cand = hypos[max_idx]
    #print("best_cand", best_cand)
    return best_cand
 


df = pd.read_csv("data/hypo_test.tsv", delimiter='\t',
                 header=None, names=['label', 'sentence'])
test_hypos = df.sentence.values.tolist()
test_label = df.label.values.tolist()
assert len(test_hypos) == len(test_label)
paras = []
for i in range(1, len(test_hypos)):
    if test_label[i-1]==1:
        paras.append(test_hypos[i].strip())


hypo_cands = get_hypo_cands(paras, topk=5) 

res=[[para, replace_hypo(para, cands)] for para, cands in tqdm(zip(paras, hypo_cands))]
df=pd.DataFrame(res)

df.to_csv("Retrieve_replace_rank_outputs.csv", index=None, header=None)


os.system("ps -ef | grep java | grep -v grep | awk '{print $2}' | xargs kill -9")
