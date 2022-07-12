from subprocess import check_output
import pandas as pd
import json
from stanfordcorenlp import StanfordCoreNLP
from tqdm import tqdm
import os
import signal
from unexpected_mask import unexpected_score

def get_all_index(n):
    res = []
    for len in range(1,n):
        res.append(list(range(i, i+len)) for i in range(n))

patterns = []
with open("pattern.tsv", 'r') as pfile:
    for line in pfile:
        pattern = line.split('\t')[0]
        patterns.append(pattern)
patterns.sort(key=lambda x:len(x.split('+')), reverse=True)

TOPN=3
nlp = StanfordCoreNLP(r'/home/zhangyunxiang/stanfordnlp_resources/stanford-corenlp-4.2.0')


punct='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~'
"""
file_path = "data/sentencdict_hypo_0508_g0.8.txt"
#file_path = "data/test_span.txt"
infile = open(file_path,'r')

lines=[]
for line in infile.readlines():
    # line = line.lower().translate(str.maketrans('', '', punct)) # 
    lines.append(line)
lines=[l.strip() for l in lines]
total_cnt=len(lines)
train_cnt=int(total_cnt*0.9)
val_cnt=total_cnt-train_cnt
print("train:",train_cnt, "val:", val_cnt)
"""
def to_outfile(lines, outfile_path, refs=None):
    outfile = open(outfile_path,'w')
    if refs is None:
        refs = lines
    assert len(lines) == len(refs)
    match_cnts=[]
    res = []
    for i,line in enumerate(tqdm(lines)):
        match_cnt=0
        match_substrs=[]
        line = line.strip().translate(str.maketrans(dict.fromkeys(punct, ' ')))
        pos_tags = nlp.pos_tag(line)
        tokens = nlp.word_tokenize(line)
        assert len(pos_tags) == len(tokens)
        tags = [t[1] for t in pos_tags]
        masked_sent2score = dict()
        maksed_span2pat = dict()
        for n in range(len(tags)-1, 0, -1):
            for start in range(len(tags)-n+1):
                pat = '+'.join(tags[start:start+n])
                if pat in patterns:
                    match_substr = ' '.join(tokens[start:start+n])
                    if True or all(match_substr not in s for s in match_substrs): # predict 细粒度 train 粗粒度
                        match_substrs.append(match_substr)
                        masked = ' '.join(['<hypo>']+tokens[:start]+['<mask>']+tokens[start+n:]).strip()
                        masked_span = ' '.join(tokens[start:start+n])
                        masked_span_score = unexpected_score(tokens, start, n)        
                        masked_sent2score[masked] = masked_span_score
                        maksed_span2pat[masked] = pat
                        res.append([line, masked_span, masked_span_score])
                        match_cnt+=1
        
        masked_sent2score_sorted = sorted(masked_sent2score,key=masked_sent2score.get,reverse=True)
        print(masked_sent2score_sorted)
        print(masked_sent2score)
        print(maksed_span2pat)
        for k in masked_sent2score_sorted[:TOPN]:
            mydict=dict()
            mydict['text']=k.strip()
            mydict['summary']=refs[i].strip()
            outfile.write(json.dumps(mydict)+'\n')

        match_cnts.append(match_cnt)
    print(outfile_path, "avg matches per sentence:", sum(match_cnts)/len(match_cnts))
    outfile.close()
    df_res = pd.DataFrame(res)
    df_res.to_csv("syntac_Span.tsv", sep='\t', header=None, index=None)

def mask_sentence(line):
    line = line.strip().translate(str.maketrans(dict.fromkeys(punct, ' ')))
    match_substrs=[]
    pos_tags = nlp.pos_tag(line)
    tokens = nlp.word_tokenize(line)
    assert len(pos_tags) == len(tokens)
    tags = [t[1] for t in pos_tags]
    masked_span2score = dict()
    for n in range(len(tags)-1, 0, -1):
        for start in range(len(tags)-n+1):
            pat = '+'.join(tags[start:start+n])
            if pat in patterns:
                match_substr = ' '.join(tokens[start:start+n])
                masked = ' '.join(tokens[:start]+['<mask>']+tokens[start+n:]).strip()
                masked_span = ' '.join(tokens[start:start+n])
                masked_span_score = unexpected_score(tokens, start, n)        
                masked_span2score[masked_span] = masked_span_score

    masked_span2score_sorted = sorted(masked_span2score,key=masked_span2score.get,reverse=True)
    masked_spans = [k.strip() for k in masked_span2score_sorted[:TOPN]]
    return masked_spans

def get_masked_spans_syntax(line):
    line = line.strip().translate(str.maketrans(dict.fromkeys(punct, ' ')))
    match_substrs=[]
    #print("get_masked_spans_syntax", line)
    pos_tags = nlp.pos_tag(line)
    tokens = nlp.word_tokenize(line)
    assert len(pos_tags) == len(tokens)
    tags = [t[1] for t in pos_tags]
    masked_span2score = dict()
    masked_span2syntax = dict()
    for n in range(len(tags)-1, 0, -1):
        for start in range(len(tags)-n+1):
            pat = '+'.join(tags[start:start+n])
            if pat in patterns:
                match_substr = ' '.join(tokens[start:start+n])
                masked = ' '.join(tokens[:start]+['<mask>']+tokens[start+n:]).strip()
                masked_span = ' '.join(tokens[start:start+n])
                masked_span_score = unexpected_score(tokens, start, n)        
                masked_span2score[masked_span] = masked_span_score
                masked_span2syntax[masked_span] = pat

    masked_span2score_sorted = sorted(masked_span2score,key=masked_span2score.get,reverse=True)
    masked_spans_syntax = [(k.strip(), masked_span2syntax[k]) for k in masked_span2score_sorted[:TOPN]]
    return masked_spans_syntax


def search_for_syntax(line, syntax):
    line = line.strip().translate(str.maketrans(dict.fromkeys(punct, ' ')))
    #print("search_for_syntax", line, syntax)
    pos_tags = nlp.pos_tag(line)
    tokens = nlp.word_tokenize(line)
    assert len(pos_tags) == len(tokens)
    tags = [t[1] for t in pos_tags]
    masked_span2score = dict()
    masked_span2syntax = dict()
    n = len(syntax.split('+'))
    results=[]
    for start in range(len(tags)-n+1):
        pat = '+'.join(tags[start:start+n])
        if pat == syntax:
            left = ' '.join(tokens[:start])
            right = ' '.join(tokens[start+n:])
            masked_span = ' '.join(tokens[start:start+n])
            results.append((left, masked_span, right))
    return results

to_outfile(["Her bounty was as infinite as the sea."], "data/debug.json")
#to_outfile(lines[:train_cnt],"data/sentencedict_train_0508_tag_top"+str(TOPN)+".json")
#to_outfile(lines[train_cnt:],"data/sentencedict_dev_0508_tag_top"+str(TOPN)+".json")

#to_outfile(lines, "data/nouse.json")
#infile.close()
"""
fIn = open("test_sentences.txt", 'r')
lines = [l for l in fIn.readlines()]
to_outfile(lines, "data/test_sentences.json")
"""

"""
df = pd.read_csv("data/hypo_dev.tsv", delimiter='\t',
                 header=None, names=['label', 'sentence'])
df = df[df['label']==1]
dev_hypos = df.sentence.values.tolist()
dev_hypos = [h.strip() for h in dev_hypos]
df = pd.read_csv("data/hypo_train.tsv", delimiter='\t',
                 header=None, names=['label', 'sentence'])
df = df[df['label']==1]
train_hypos = df.sentence.values.tolist()
train_hypos = [h.strip() for h in train_hypos]

to_outfile(dev_hypos, "data/hypo_dev.json") # gold(hypos) cannot leak!
to_outfile(train_hypos, "data/hypo_train.json") # gold(hypos) cannot leak!
"""
"""
with open("data/sentencedict_cleaned_dup.txt", 'r') as inF:
    sentencedict = [l.strip() for l in inF.readlines()]
to_outfile(sentencedict, "data/sentencedict_train.json")
"""


"""
df = pd.read_csv("HYPO.tsv", delimiter='\t',
                 header=None, names=['hypo', 'para', 'mnc'])
df = df.dropna()
print('has Number of sentences: {:,}\n'.format(df.shape[0]))
paras = df.para.values.tolist()
hypos = df.hypo.values.tolist()

assert len(paras) == len(hypos)

# to_outfile(paras, "data/test.json", hypos)
# to_outfile(hypos, "data/test_tag_nouse.json", hypos)
to_outfile(paras, "data/test_tag.json", paras) # gold(hypos) cannot leak!
"""
#os.system("ps -ef | grep java | grep -v grep | awk '{print $2}' | xargs kill -9")
