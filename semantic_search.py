"""
This script contains an example how to perform semantic search with PyTorch. It performs exact nearest neighborh search.

As dataset, we use the Quora Duplicate Questions dataset, which contains about 500k questions (we only use about 100k):
https://www.quora.com/q/quoradata/First-Quora-Dataset-Release-Question-Pairs


As embeddings model, we use the SBERT model 'quora-distilbert-multilingual',
that it aligned for 100 languages. I.e., you can type in a question in various languages and it will
return the closest questions in the corpus (questions in the corpus are mainly in English).


Google Colab example: https://colab.research.google.com/drive/12cn5Oo0v3HfQQ8Tv6-ukgxXSmT3zl35A?usp=sharing
"""
from sentence_transformers import SentenceTransformer, util
import os
import csv
import pickle
import time


# model_name = 'quora-distilbert-multilingual'
model_name = 'model_save/hypo-pair-paraphrase-distilroberta-base-v1'
model = SentenceTransformer(model_name)
dataset_path = "data/sentencdict_hypo_0508_g0.8.txt"


# embedding_cache_path = 'quora-embeddings-{}-size-{}.pkl'.format(model_name.replace('/', '_'), max_corpus_size)
embedding_cache_path = 'HYPO-Large-embeddings.pkl'


def get_hypo_cands(literal_sentences, topk=10):

    #Check if embedding cache path exists
    if not os.path.exists(embedding_cache_path):
        # Check if the dataset exists. If not, download and extract
        # Download dataset if needed

        # Get all unique sentences from the file
        with open(dataset_path, encoding='utf8') as fIn:
            corpus_sentences = [l.strip() for l in fIn.readlines()]
        print("Encode the corpus. This might take a while")
        corpus_embeddings = model.encode(corpus_sentences, show_progress_bar=True, convert_to_tensor=True)

        print("Store file on disc")
        with open(embedding_cache_path, "wb") as fOut:
            pickle.dump({'sentences': corpus_sentences, 'embeddings': corpus_embeddings}, fOut)
    else:
        print("Load pre-computed embeddings from disc")
        with open(embedding_cache_path, "rb") as fIn:
            cache_data = pickle.load(fIn)
            corpus_sentences = cache_data['sentences']
            corpus_embeddings = cache_data['embeddings']

    ###############################
    print("Corpus loaded with {} sentences / embeddings".format(len(corpus_sentences)))

    #Move embeddings to the target device of the model
    corpus_embeddings = corpus_embeddings.to('cuda')


    
    print("Encoding literal queries sentences...")
    query_embeddings = model.encode(literal_sentences, show_progress_bar=True, convert_to_tensor=True)

    # corpus_embeddings = corpus_embeddings.to('cuda')
    corpus_embeddings = util.normalize_embeddings(corpus_embeddings)

    query_embeddings = query_embeddings.to('cuda')
    query_embeddings = util.normalize_embeddings(query_embeddings)

    hits = util.semantic_search(query_embeddings, corpus_embeddings, top_k=topk, score_function=util.dot_score)

    res=[[corpus_sentences[h['corpus_id']] for h in hit] for hit in hits]
    return res
