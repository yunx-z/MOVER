import random
import pandas as pd
f=open("HYPO.tsv",'r')
all_lines=f.read().split('\n')
print("total line",len(all_lines))
random.shuffle(all_lines)

train_cnt=int(len(all_lines)*0.8)
dev_cnt=int(len(all_lines)*0.1)
train=all_lines[:train_cnt]
dev=all_lines[train_cnt:train_cnt+dev_cnt]
test=all_lines[train_cnt+dev_cnt:]
print("train",len(train),"dev",len(dev),"test",len(test))

res=[]
for line in train:
    sents=line.split('\t')
    sents=[s.strip() for s in sents]
    res.append([1,sents[0]])
    if len(sents)>1 and sents[1].strip()!="":
        res.append([0,sents[1]])
    if len(sents)>2 and sents[1].strip()!="":
        res.append([0,sents[2]])
res=pd.DataFrame(res)
res.to_csv("./data/hypo_train.tsv",sep='\t',header=None,index=None)


res=[]
for line in dev:
    sents=line.split('\t')
    sents=[s.strip() for s in sents]
    res.append([1,sents[0]])
    if len(sents)>1 and sents[1].strip()!="":
        res.append([0,sents[1]])
    if len(sents)>2 and sents[1].strip()!="":
        res.append([0,sents[2]])

res=pd.DataFrame(res)
res.to_csv("./data/hypo_dev.tsv",sep='\t',header=None,index=None)
res=[]
for line in test:
    sents=line.split('\t')
    sents=[s.strip() for s in sents]
    res.append([1,sents[0]])
    if len(sents)>1 and sents[1].strip()!="":
        res.append([0,sents[1]])
    if len(sents)>2 and sents[1].strip()!="":
        res.append([0,sents[2]])


res=pd.DataFrame(res)
res.to_csv("./data/hypo_test.tsv",sep='\t',header=None,index=None)
