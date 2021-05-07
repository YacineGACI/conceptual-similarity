from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error # For evaluating the trained model
import pickle
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from nltk.tokenize import word_tokenize

import nltk
from nltk.metrics import edit_distance
from nltk.metrics import jaccard_distance
from nltk.translate.bleu_score import modified_precision, SmoothingFunction
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial import distance
from scipy.stats import pearsonr, spearmanr, kendalltau, entropy
from sklearn.metrics import mutual_info_score
from sklearn.metrics.pairwise import sigmoid_kernel, pairwise_kernels


#################################   Features   ##########################################
#########################################################################################
#########################################################################################

##############  n-gram distance features  ##############

def bleu_score(s1, s2):
    hypothesis = s2.split(' ')
    reference = s1.split(' ')
    cc = SmoothingFunction()
    return nltk.translate.bleu_score.sentence_bleu([reference], hypothesis, smoothing_function=cc.method4)


def levenshtein_distance(s1, s2):
    return edit_distance(s1, s2)


def jaccard_distance_word(s1, s2):
    w1 = set(s1.split(" "))
    w2 = set(s2.split(" "))
    return jaccard_distance(w1, w2)


def jaccard_distance_char(s1, s2):
    w1 = set(s1)
    w2 = set(s2)
    return jaccard_distance(w1, w2)


def ngram_overlap(s1, s2, n=1):
    w1 = s1.split(" ")
    w2 = s2.split(" ")
    return float(modified_precision([w1], w2, n))


def dice_index(s1, s2): 
    a = set(s1.split()) 
    b = set(s2.split())
    c = a.intersection(b)
    return 2*float(len(c)) / (len(a) + len(b))


def overlap_index(s1, s2): 
    a = set(s1.split()) 
    b = set(s2.split())
    c = a.intersection(b)
    return float(len(c)) / min(len(a) , len(b) )


##############  linear kernel features  ##############

def cosine_sim(u, v):
    u = u.reshape(1, u.shape[0])
    v = v.reshape(1, v.shape[0])
    return cosine_similarity(u, v)[0][0]


def euclidean_distance(u, v):
    return np.sqrt(np.dot(u-v, u-v))


def manhattan_distance(u, v):
    return distance.cityblock(u, v)



##############  statistical measures features  ##############

def pearson_correlation(u, v):
    return pearsonr(u, v)[0]


def spearman_correlation(u, v):
    return spearmanr(u, v)[0]


def kendall_tau(u, v):
    return kendalltau(u, v)[0]


##############  probabilistic measures features  ##############

def kl_divergence(u, v):
    return entropy(u, v, base=2)


def mutual_info(u, v):
    return mutual_info_score(u, v)


##############  kernel features  ##############

def sigmoid_kernel(u, v):
    return sigmoid_kernel(np.array(u).reshape(1, -1), np.array(v).reshape(1, -1))[0][0]


def kernel(u, v, metric="linear"):
    return pairwise_kernels(np.array(u).reshape(1, -1), np.array(v).reshape(1, -1), metric=metric)[0][0]


#########################################################################################
#########################################################################################
#########################################################################################




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





def get_features(s1, s2):

    u_w2v = word2vec_embedding(s1)
    v_w2v = word2vec_embedding(s2)

    u_glove = glove_embedding(s1)
    v_glove = glove_embedding(s2)

    u_paragram = paragram_embedding(s1)
    v_paragram = paragram_embedding(s2)

    new_features = []
    new_features.append(bleu_score(s1, s2))
    new_features.append(levenshtein_distance(s1, s2))
    new_features.append(jaccard_distance_word(s1, s2))
    new_features.append(jaccard_distance_char(s1, s2))
    new_features.append(ngram_overlap(s1, s2, 1))
    new_features.append(ngram_overlap(s1, s2, 2))
    new_features.append(ngram_overlap(s1, s2, 3))
    new_features.append(ngram_overlap(s1, s2, 4))
    new_features.append(dice_index(s1, s2))
    new_features.append(overlap_index(s1, s2))

    new_features.append(cosine_sim(u_w2v, v_w2v))
    new_features.append(cosine_sim(u_glove, v_glove))
    new_features.append(cosine_sim(u_paragram, v_paragram))
    new_features.append(euclidean_distance(u_w2v, v_w2v))
    new_features.append(euclidean_distance(u_glove, v_glove))
    new_features.append(euclidean_distance(u_paragram, v_paragram))
    new_features.append(manhattan_distance(u_w2v, v_w2v))
    new_features.append(manhattan_distance(u_glove, v_glove))
    new_features.append(manhattan_distance(u_paragram, v_paragram))

    new_features.append(pearson_correlation(u_w2v, v_w2v))
    new_features.append(pearson_correlation(u_glove, v_glove))
    new_features.append(pearson_correlation(u_paragram, v_paragram))
    new_features.append(spearman_correlation(u_w2v, v_w2v))
    new_features.append(spearman_correlation(u_glove, v_glove))
    new_features.append(spearman_correlation(u_paragram, v_paragram))
    new_features.append(kendall_tau(u_w2v, v_w2v))
    new_features.append(kendall_tau(u_glove, v_glove))
    new_features.append(kendall_tau(u_paragram, v_paragram))

    new_features.append(mutual_info(u_w2v, v_w2v))
    new_features.append(mutual_info(u_glove, v_glove))
    new_features.append(mutual_info(u_paragram, v_paragram))

    new_features.append(kernel(u_w2v, v_w2v, metric="laplacian"))
    new_features.append(kernel(u_glove, v_glove, metric="laplacian"))
    new_features.append(kernel(u_paragram, v_paragram, metric="laplacian"))
    new_features.append(kernel(u_w2v, v_w2v, metric="sigmoid"))
    new_features.append(kernel(u_glove, v_glove, metric="sigmoid"))
    new_features.append(kernel(u_paragram, v_paragram, metric="sigmoid"))
    new_features.append(kernel(u_w2v, v_w2v, metric="rbf"))
    new_features.append(kernel(u_glove, v_glove, metric="rbf"))
    new_features.append(kernel(u_paragram, v_paragram, metric="rbf"))
    new_features.append(kernel(u_w2v, v_w2v, metric="polynomial"))
    new_features.append(kernel(u_glove, v_glove, metric="polynomial"))
    new_features.append(kernel(u_paragram, v_paragram, metric="polynomial"))

    return new_features



w2vec_model =  Word2Vec.load("embeddings/word2vec_sg_100_5_5.model").wv
glove_model = KeyedVectors.load_word2vec_format("embeddings/glove.txt", binary=False)
with open("embeddings/paragram_embed_ws353.pkl", 'rb') as f:
    paragram_model = pickle.load(f)


if __name__ == "__main__":

    print("Word vectors loaded")

    phrases, labels = read_training_data("dataset/generated/sim_dataset_no_expansion_51_10000.csv")
    # phrases, labels = read_training_data("data/training_data_noisy_0.1.csv")
    num_training_examples = len(phrases)
    print('Training dataset read')



    train_test_split = 0.7

    x_train = []
    for s1, s2 in phrases:
        x_train.append(get_features(s1, s2))
    print("Data Processing done")

    x_test = x_train[int(num_training_examples * train_test_split):]
    y_test = labels[int(num_training_examples * train_test_split):]

    x_train = x_train[:int(num_training_examples * train_test_split)]
    y_train = labels[:int(num_training_examples * train_test_split)]


    hyperparameters = {
        'n_estimators': 100,
        'max_depth': 10,
        'random_state': 0
    }

    rf_model = RandomForestRegressor(n_estimators=hyperparameters['n_estimators'], max_depth=hyperparameters['max_depth'], random_state=hyperparameters['random_state'])
    rf_model.fit(x_train, y_train)

    print("Training complete")

    # Saving the trained models
    with open("models/random_forest.pkl", "wb") as f:
        pickle.dump({
            'model': rf_model,
            'hyperparameters': hyperparameters
        }, f)

    # Evaluate the trained model
    y_pred_rf = rf_model.predict(x_test)

    print("MSE --> ", mean_squared_error(y_test, y_pred_rf))
    print("MAE --> ", mean_absolute_error(y_test, y_pred_rf))
