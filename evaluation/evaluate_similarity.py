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

from models import BERT_classification, BERT_AGG_MATCH, Siamese
from training.train_random_forest import get_features, w2vec_model, glove_model, paragram_model


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





def compute_similarity_bert_classification(s1, s2):
    tmp = tokenizer.encode_plus(s1, s2, add_special_tokens=True, padding="max_length", max_length=20)
    input_ids = torch.tensor(tmp['input_ids']).unsqueeze(0)
    attention_mask = torch.tensor(tmp['attention_mask']).unsqueeze(0)
    seg = torch.tensor(tmp['token_type_ids']).unsqueeze(0)

    output = model(input_ids, attention_mask, seg)
    return output.item()




def compute_similarity_bert_agg_match(s1, s2):
    tmp = tokenizer.encode_plus(s1, s2, add_special_tokens=True, padding="max_length", max_length=20)
    input_ids = torch.tensor(tmp['input_ids']).unsqueeze(0)
    attention_mask = torch.tensor(tmp['attention_mask']).unsqueeze(0)
    seg = torch.tensor(tmp['token_type_ids']).unsqueeze(0)
    h_0 = torch.rand(2, input_ids.shape[0], hyperparameters['lstm_hidden_size'])
    c_0 = torch.rand(2, input_ids.shape[0], hyperparameters['lstm_hidden_size'])

    output = model(input_ids, attention_mask, seg, h_0, c_0)
    return output.item()




def compute_similarity_bert_cosine(s1, s2, pooling='cls'):
    def cls_pooling(tensor):
        return tensor[:,0,:]

    def sum_pooling(tensor):
        return torch.sum(tensor, dim=1)

    def mean_pooling(tensor):
        return sum_pooling(tensor) / tensor.shape[1]

    def max_pooling(tensor):
        return torch.max(tensor, dim=1).values

    s1 = tokenizer.encode_plus(s1, add_special_tokens=True)
    s2 = tokenizer.encode_plus(s2, add_special_tokens=True)
    input_1 = torch.tensor(s1['input_ids']).unsqueeze(0)
    input_2 = torch.tensor(s2['input_ids']).unsqueeze(0)

    output_1 = model(input_1)[0]
    output_2 = model(input_2)[0]

    if pooling == 'cls':
        pooling_strategy = cls_pooling
    if pooling == 'sum':
        pooling_strategy = sum_pooling
    if pooling == 'mean':
        pooling_strategy = mean_pooling
    if pooling == 'max':
        pooling_strategy = max_pooling

    output_1 = pooling_strategy(output_1).detach().numpy()
    output_2 = pooling_strategy(output_2).detach().numpy()
    
    return cosine_similarity(output_1, output_2)[0][0]


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



def paragram_embedding_unpooled(phrase):
    return [paragram_model[w] if w in paragram_model.keys() else [0] * hyperparameters['embedding_dim'] for w in word_tokenize(phrase.lower())]




def compute_similarity_siamese(s1, s2):
    input_1 = torch.tensor(paragram_embedding_unpooled(s1)).unsqueeze(0)
    input_2 = torch.tensor(paragram_embedding_unpooled(s2)).unsqueeze(0)

    h_0 = torch.rand(1, input_1.shape[0], hyperparameters['gru_hidden_size'])

    output = model(input_1, input_2, h_0)
    return output.item()




def merge_predictions(predictions, weights):
    new_predictions = []
    for i in range(len(predictions[0])):
        mean_pred = 0
        for p, w in zip(predictions, weights):
            mean_pred += p[i] * w
        mean_pred /= sum(weights)
        new_predictions.append(mean_pred)
    return new_predictions




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
    
    # Load the similarity model

    ######################      Random Forest Model       #####################

    with open("models/random_forest.pkl", "rb") as f:
        rf_model = pickle.load(f)["model"]

    rf_predictions = []
    for s1, s2 in sentences:
        rf_predictions.append(rf_model.predict(np.array(get_features(s1, s2)).reshape(1, -1))[0])

    assert len(rf_predictions) == len(labels)

    print("Random Forest model")
    print("Pearson correlation: {}".format(pearsonr(rf_predictions, labels)[0]))
    print("Spearman correlation: {}".format(spearmanr(rf_predictions, labels)[0]))
    print("Kendall correlation: {}".format(kendalltau(rf_predictions, labels)[0]))
    print("Mean Absolute Error: {}".format(mean_absolute_error(rf_predictions, labels)))
    print()



    ####################  BERT AGG MATCH  ##########################
    saved = torch.load("models/bert_agg_match.pt")
    hyperparameters = saved["hyperparameters"]
    model = BERT_AGG_MATCH(bert_hidden_size=hyperparameters['bert_hidden_size'], lstm_hidden_size=hyperparameters['lstm_hidden_size'], classifier_hidden_size=hyperparameters['classifier_hidden_size'], dropout=hyperparameters['dropout'], pooling=hyperparameters['pooling'])
    model.load_state_dict(saved['model'])
    model.eval()
    tokenizer = model.tokenizer

    bert_agg_match_predictions = []
    for s1, s2 in sentences:
        bert_agg_match_predictions.append(compute_similarity_bert_agg_match(s1, s2))

    assert len(bert_agg_match_predictions) == len(labels)

    print("BERT AGG MATCH model")
    print("Pearson correlation: {}".format(pearsonr(bert_agg_match_predictions, labels)[0]))
    print("Spearman correlation: {}".format(spearmanr(bert_agg_match_predictions, labels)[0]))
    print("Kendall correlation: {}".format(kendalltau(bert_agg_match_predictions, labels)[0]))
    print("Mean Absolute Error: {}".format(mean_absolute_error(bert_agg_match_predictions, labels)))
    print()



    ################  Siamese Model  ##################
    saved = torch.load("models/siamese.pt")
    hyperparameters = saved["hyperparameters"]
    model = Siamese(embedding_size=hyperparameters['embedding_dim'], gru_hidden_size=hyperparameters['gru_hidden_size'], classifier_hidden_size=hyperparameters['classifier_hidden_size'], dropout=hyperparameters['dropout'])
    model.load_state_dict(saved['model'])
    model.eval()
    
    with open("embeddings/paragram_embed_ws353.pkl", 'rb') as f:
        paragram_model = pickle.load(f)


    siamese_predictions = []
    for s1, s2 in sentences:
        siamese_predictions.append(compute_similarity_siamese(s1, s2))

    assert len(siamese_predictions) == len(labels)

    print("Siamese model")
    print("Pearson correlation: {}".format(pearsonr(siamese_predictions, labels)[0]))
    print("Spearman correlation: {}".format(spearmanr(siamese_predictions, labels)[0]))
    print("Kendall correlation: {}".format(kendalltau(siamese_predictions, labels)[0]))
    print("Mean Absolute Error: {}".format(mean_absolute_error(siamese_predictions, labels)))
    print()



    ##############   Paragram Embedding   ##############
    paragram_predictions = []
    for s1, s2 in sentences:
        paragram_predictions.append(cosine_sim(s1, s2, embedding="paragram"))

    assert len(paragram_predictions) == len(labels)

    print("Cosine Paragram model")
    print("Pearson correlation: {}".format(pearsonr(paragram_predictions, labels)[0]))
    print("Spearman correlation: {}".format(spearmanr(paragram_predictions, labels)[0]))
    print("Kendall correlation: {}".format(kendalltau(paragram_predictions, labels)[0]))
    print("Mean Absolute Error: {}".format(mean_absolute_error(paragram_predictions, labels)))
    print()



    #########   Ensemble Model   ############
    all_predictions = [bert_agg_match_predictions, siamese_predictions, paragram_predictions, rf_predictions]
    weights = [10, 3, 3, 1]
    ensemble_predictions = merge_predictions(all_predictions, weights)

    assert len(ensemble_predictions) == len(labels)

    print("Ensemble model")
    print("Pearson correlation: {}".format(pearsonr(ensemble_predictions, labels)[0]))
    print("Spearman correlation: {}".format(spearmanr(ensemble_predictions, labels)[0]))
    print("Kendall correlation: {}".format(kendalltau(ensemble_predictions, labels)[0]))
    print("Mean Absolute Error: {}".format(mean_absolute_error(ensemble_predictions, labels)))
    print()


    ######################      BERT Classification     #######################
    saved = torch.load("models/bert_classification.pt")
    hyperparameters = saved["hyperparameters"]
    model = BERT_classification(bert_hidden_size=hyperparameters['bert_hidden_size'], classifier_hidden_size=hyperparameters['classifier_hidden_size'], dropout=hyperparameters['dropout'])
    model.load_state_dict(saved['model'])
    model.eval()
    tokenizer = model.tokenizer

    predictions = []
    for s1, s2 in sentences:
        predictions.append(compute_similarity_bert_classification(s1, s2))

    assert len(predictions) == len(labels)

    print("BERT Classification model")
    print("Pearson correlation: {}".format(pearsonr(predictions, labels)[0]))
    print("Spearman correlation: {}".format(spearmanr(predictions, labels)[0]))
    print("Kendall correlation: {}".format(kendalltau(predictions, labels)[0]))
    print("Mean Absolute Error: {}".format(mean_absolute_error(predictions, labels)))
    print()



    #############################   BERT Cosine   ######################
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')


    predictions = []
    for s1, s2 in sentences:
        predictions.append(compute_similarity_bert_cosine(s1, s2, 'cls'))

    assert len(predictions) == len(labels)

    print("BERT Cosine CLS model")
    print("Pearson correlation: {}".format(pearsonr(predictions, labels)[0]))
    print("Spearman correlation: {}".format(spearmanr(predictions, labels)[0]))
    print("Kendall correlation: {}".format(kendalltau(predictions, labels)[0]))
    print("Mean Absolute Error: {}".format(mean_absolute_error(predictions, labels)))
    print()

    predictions = []
    for s1, s2 in sentences:
        predictions.append(compute_similarity_bert_cosine(s1, s2, 'sum'))

    assert len(predictions) == len(labels)

    print("BERT Cosine SUM model")
    print("Pearson correlation: {}".format(pearsonr(predictions, labels)[0]))
    print("Spearman correlation: {}".format(spearmanr(predictions, labels)[0]))
    print("Kendall correlation: {}".format(kendalltau(predictions, labels)[0]))
    print("Mean Absolute Error: {}".format(mean_absolute_error(predictions, labels)))
    print()

    predictions = []
    for s1, s2 in sentences:
        predictions.append(compute_similarity_bert_cosine(s1, s2, 'mean'))

    assert len(predictions) == len(labels)

    print("BERT Cosine MEAN model")
    print("Pearson correlation: {}".format(pearsonr(predictions, labels)[0]))
    print("Spearman correlation: {}".format(spearmanr(predictions, labels)[0]))
    print("Kendall correlation: {}".format(kendalltau(predictions, labels)[0]))
    print("Mean Absolute Error: {}".format(mean_absolute_error(predictions, labels)))
    print()

    predictions = []
    for s1, s2 in sentences:
        predictions.append(compute_similarity_bert_cosine(s1, s2, 'max'))

    assert len(predictions) == len(labels)

    print("BERT Cosine MAX model")
    print("Pearson correlation: {}".format(pearsonr(predictions, labels)[0]))
    print("Spearman correlation: {}".format(spearmanr(predictions, labels)[0]))
    print("Kendall correlation: {}".format(kendalltau(predictions, labels)[0]))
    print("Mean Absolute Error: {}".format(mean_absolute_error(predictions, labels)))
    print()

