import pickle

from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
import numpy as np
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from transformers import BertModel, BertTokenizer
import torch
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import Birch
from sklearn.cluster import SpectralClustering


def read_training_data(filename):
    phrases = []
    labels = []
    with open(filename, 'r') as f:
        for l in f.readlines():
            s1, s2, sim = l.split(',')
            phrases.append((s1.strip(' '), s2.strip(' ')))
            labels.append(float(sim.strip('\n ')))
    return phrases, labels



def word2vec_embedding(phrase):
    return np.mean([w2vec_model[w] for w in phrase.split(' ') if w in w2vec_model.vocab], axis=0)

def glove_embedding(phrase):
    return np.mean([glove_model[w] for w in phrase.split(' ') if w in glove_model.wv.vocab], axis=0)

def paragram_embedding(phrase):
    return np.mean([paragram_model[w] for w in word_tokenize(phrase.lower()) if w in paragram_model.keys()], axis=0)


def clustering_accuracy(ground_truth, predictions):
    # Compute the accuracy as is
    accuracy_1 = accuracy_score(ground_truth, predictions)

    # Swap the labels of predictions in case the cluster labels are not in line with the ground truth
    indices_one = predictions == 1
    indices_zero = predictions == 0
    predictions[indices_one] = 0 # replacing 1s with 0s
    predictions[indices_zero] = 1 # replacing 0s with 1s

    accuracy_2 = accuracy_score(ground_truth, predictions)
    return max(accuracy_1, accuracy_2)




if __name__ == "__main__":
    phrases, labels = read_training_data("dataset/generated/sim_dataset_no_expansion_51_10000.csv")
    assert len(phrases) == len(labels)
    num_training_example = len(phrases)
    print('Training dataset read')

    
    w2vec_model =  Word2Vec.load("embeddings/word2vec_sg_100_5_5.model").wv
    glove_model = KeyedVectors.load_word2vec_format("embeddings/glove.txt", binary=False)
    with open("embeddings/paragram_embed_ws353.pkl", 'rb') as f:
        paragram_model = pickle.load(f)
    
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # bert_model = BertModel.from_pretrained('bert-base-uncased')

    paragram_vectors = []
    w2v_vectors = []
    glove_vectors = []
    bert_vectors = []
    for s1, s2 in phrases:
        s1_paragram = paragram_embedding(s1)
        s2_paragram = paragram_embedding(s2)
        paragram_vectors.append(np.concatenate((s1_paragram, s2_paragram)))

        s1_w2v = word2vec_embedding(s1)
        s2_w2v = word2vec_embedding(s2)
        w2v_vectors.append(np.concatenate((s1_w2v, s2_w2v)))

        s1_glove = glove_embedding(s1)
        s2_glove = glove_embedding(s2)
        glove_vectors.append(np.concatenate((s1_glove, s2_glove)))

        # input = torch.tensor(tokenizer.encode_plus(s1, s2, add_special_tokens=True)['input_ids']).unsqueeze(0)
        # bert_vectors.append(bert_model(input)[0][:,0,:])



    
    # KMeans
    print("KMeans")
    vectors = np.array(paragram_vectors)
    kmeans = KMeans(n_clusters=2, random_state=0).fit(vectors)
    predictions = kmeans.labels_
    kmeans_accuracy = clustering_accuracy(labels, predictions)
    print("Paragram: {}".format(kmeans_accuracy))

    vectors = np.array(w2v_vectors)
    kmeans = KMeans(n_clusters=2, random_state=0).fit(vectors)
    predictions = kmeans.labels_
    kmeans_accuracy = clustering_accuracy(labels, predictions)
    print("W2v: {}".format(kmeans_accuracy))

    vectors = np.array(glove_vectors)
    kmeans = KMeans(n_clusters=2, random_state=0).fit(vectors)
    predictions = kmeans.labels_
    kmeans_accuracy = clustering_accuracy(labels, predictions)
    print("Glove: {}".format(kmeans_accuracy))

    print()

  

    # Agglomerative Clustering
    print("Agglomerative Clustering")
    vectors = np.array(paragram_vectors)
    agglo = AgglomerativeClustering(n_clusters=2).fit(vectors)
    predictions = agglo.labels_
    kmeans_accuracy = clustering_accuracy(labels, predictions)
    print("Paragram: {}".format(kmeans_accuracy))

    vectors = np.array(w2v_vectors)
    agglo = AgglomerativeClustering(n_clusters=2).fit(vectors)
    predictions = agglo.labels_
    kmeans_accuracy = clustering_accuracy(labels, predictions)
    print("W2v: {}".format(kmeans_accuracy))

    vectors = np.array(glove_vectors)
    agglo = AgglomerativeClustering(n_clusters=2).fit(vectors)
    predictions = agglo.labels_
    kmeans_accuracy = clustering_accuracy(labels, predictions)
    print("Glove: {}".format(kmeans_accuracy))

    print()



    # Birch
    print("Birch")
    vectors = np.array(paragram_vectors)
    agglo = Birch(n_clusters=2, threshold=0.01).fit(vectors)
    predictions = agglo.labels_
    kmeans_accuracy = clustering_accuracy(labels, predictions)
    print("Paragram: {}".format(kmeans_accuracy))

    vectors = np.array(w2v_vectors)
    agglo = Birch(n_clusters=2, threshold=0.01).fit(vectors)
    predictions = agglo.labels_
    kmeans_accuracy = clustering_accuracy(labels, predictions)
    print("W2v: {}".format(kmeans_accuracy))

    vectors = np.array(glove_vectors)
    agglo = Birch(n_clusters=2, threshold=0.01).fit(vectors)
    predictions = agglo.labels_
    kmeans_accuracy = clustering_accuracy(labels, predictions)
    print("Glove: {}".format(kmeans_accuracy))
    
    print()

