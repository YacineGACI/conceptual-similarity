from os import listdir
from os.path import isfile, join
import pickle

from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr, spearmanr, kendalltau
from nltk.tokenize import word_tokenize
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch 
from transformers import BertModel, BertTokenizer
from gensim.models import Word2Vec
from gensim.models import KeyedVectors


def get_files_in_folder(folder):
    return [join(folder, f) for f in listdir(folder) if isfile(join(folder, f))]


def read_eval_data(filename):
    sentences, labels = [], []
    with open(filename, 'r') as f:
        for line in f.readlines():
            s1, s2, l = line.strip('\n ').split(',')
            sentences.append((s1, s2))
            labels.append(int(l))

    return sentences, labels


def merge_labels(labels):
    new_labels = []
    for i in range(len(labels[0])):
        tmp = []
        for l in labels:
            tmp.append(l[i])
        merged_value = sum(tmp) / len(tmp)
        new_labels.append(merged_value)
    return new_labels


def cosine_sim(s1, s2, embedding="w2v"):
    if embedding == "w2v":
        embedding_function = word2vec_embedding
    if embedding == "glove":
        embedding_function = glove_embedding
    if embedding == "paragram":
        embedding_function = paragram_embedding
    if "meta" in embedding:
        embedding_function = meta_embedding

    s1 = embedding_function(s1)
    s2 = embedding_function(s2)

    s1 = s1.reshape(1, s1.shape[0])
    s2 = s2.reshape(1, s2.shape[0])

    return cosine_similarity(s1, s2)[0][0]
    


def word2vec_embedding(phrase):
    return np.mean([w2vec_model[w] for w in phrase.split(' ') if w in w2vec_model.vocab], axis=0)

def glove_embedding(phrase):
    return np.mean([glove_model[w] for w in phrase.split(' ') if w in glove_model.wv.vocab], axis=0)

def paragram_embedding(phrase):
    return np.mean([paragram_model[w] for w in word_tokenize(phrase.lower()) if w in paragram_model.keys()], axis=0)

def meta_embedding(phrase):
    return np.mean([meta_model[w] for w in word_tokenize(phrase.lower()) if w in meta_model.keys()], axis=0)




if __name__ == "__main__":

    # Find all evaluation files
    eval_directory = "evaluation/data/participants"
    eval_files = get_files_in_folder(eval_directory)
    
    # Load corresponding evaluation data
    data = [read_eval_data(f) for f in eval_files]
    sentences = data[0][0]
    labels = [x[1] for x in data]

    # Merge the labels from different workers
    labels = [x/5 for x in merge_labels(labels)]



    ######################    Cosine Similarity     ##############
    w2vec_model =  Word2Vec.load("embeddings/word2vec_sg_100_5_5.model").wv
    glove_model = KeyedVectors.load_word2vec_format("embeddings/glove.txt", binary=False)
    with open("embeddings/paragram_embed_ws353.pkl", 'rb') as f:
        paragram_model = pickle.load(f)

    predictions = []
    for s1, s2 in sentences:
        predictions.append(cosine_sim(s1, s2, embedding="w2v"))

    assert len(predictions) == len(labels)

    print("Cosine W2V model")
    print("Pearson correlation: {}".format(pearsonr(predictions, labels)[0]))
    print("Spearman correlation: {}".format(spearmanr(predictions, labels)[0]))
    print("Kendall correlation: {}".format(kendalltau(predictions, labels)[0]))
    print("Mean Absolute Error: {}".format(mean_absolute_error(predictions, labels)))
    print()


    predictions = []
    for s1, s2 in sentences:
        predictions.append(cosine_sim(s1, s2, embedding="glove"))

    assert len(predictions) == len(labels)

    print("Cosine Glove model")
    print("Pearson correlation: {}".format(pearsonr(predictions, labels)[0]))
    print("Spearman correlation: {}".format(spearmanr(predictions, labels)[0]))
    print("Kendall correlation: {}".format(kendalltau(predictions, labels)[0]))
    print("Mean Absolute Error: {}".format(mean_absolute_error(predictions, labels)))
    print()


    predictions = []
    for s1, s2 in sentences:
        predictions.append(cosine_sim(s1, s2, embedding="paragram"))

    assert len(predictions) == len(labels)

    print("Cosine Paragram model")
    print("Pearson correlation: {}".format(pearsonr(predictions, labels)[0]))
    print("Spearman correlation: {}".format(spearmanr(predictions, labels)[0]))
    print("Kendall correlation: {}".format(kendalltau(predictions, labels)[0]))
    print("Mean Absolute Error: {}".format(mean_absolute_error(predictions, labels)))
    print()




    ##############   Cosine Meta Embeddings   #################
    with open("embeddings/meta_embedding/models/concat.pkl", 'rb') as f:
        meta_model = pickle.load(f)

    predictions = []
    for s1, s2 in sentences:
        predictions.append(cosine_sim(s1, s2, embedding="meta-concat"))

    assert len(predictions) == len(labels)

    print("Cosine Meta-Concat model")
    print("Pearson correlation: {}".format(pearsonr(predictions, labels)[0]))
    print("Spearman correlation: {}".format(spearmanr(predictions, labels)[0]))
    print("Kendall correlation: {}".format(kendalltau(predictions, labels)[0]))
    print("Mean Absolute Error: {}".format(mean_absolute_error(predictions, labels)))
    print()



    ##############   Cosine Meta Embeddings   #################
    with open("embeddings/meta_embedding/models/avg.pkl", 'rb') as f:
        meta_model = pickle.load(f)

    predictions = []
    for s1, s2 in sentences:
        predictions.append(cosine_sim(s1, s2, embedding="meta-concat"))

    assert len(predictions) == len(labels)

    print("Cosine Meta-AVG model")
    print("Pearson correlation: {}".format(pearsonr(predictions, labels)[0]))
    print("Spearman correlation: {}".format(spearmanr(predictions, labels)[0]))
    print("Kendall correlation: {}".format(kendalltau(predictions, labels)[0]))
    print("Mean Absolute Error: {}".format(mean_absolute_error(predictions, labels)))
    print()




    ############    Autoencoder Meta Embeddings  ##############
    with open("embeddings/meta_embedding/models/meta_autoencoder_aaeme.pkl", 'rb') as f:
        meta_model = pickle.load(f)

    predictions = []
    for s1, s2 in sentences:
        predictions.append(cosine_sim(s1, s2, embedding="meta-autoencoder-aaeme"))

    assert len(predictions) == len(labels)

    print("Cosine Meta-Autoencoder-AAEME model")
    print("Pearson correlation: {}".format(pearsonr(predictions, labels)[0]))
    print("Spearman correlation: {}".format(spearmanr(predictions, labels)[0]))
    print("Kendall correlation: {}".format(kendalltau(predictions, labels)[0]))
    print("Mean Absolute Error: {}".format(mean_absolute_error(predictions, labels)))
    print()