import pickle

from nltk.tokenize import word_tokenize
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch 
from transformers import BertModel, BertTokenizer
from gensim.models import Word2Vec
from gensim.models import KeyedVectors

from models import BERT_classification, BERT_AGG_MATCH, Siamese
from training.train_random_forest import get_features, w2vec_model, glove_model, paragram_model



def compute_similarity_bert_classification(s1, s2):
    tmp = bert_classification_tokenizer.encode_plus(s1, s2, add_special_tokens=True, padding="max_length", max_length=20)
    input_ids = torch.tensor(tmp['input_ids']).unsqueeze(0)
    attention_mask = torch.tensor(tmp['attention_mask']).unsqueeze(0)
    seg = torch.tensor(tmp['token_type_ids']).unsqueeze(0)

    output = bert_classification_model(input_ids, attention_mask, seg)
    return output.item()





def compute_similarity_bert_agg_match(s1, s2):
    tmp = bert_lstm_tokenizer.encode_plus(s1, s2, add_special_tokens=True, padding="max_length", max_length=20)
    input_ids = torch.tensor(tmp['input_ids']).unsqueeze(0)
    attention_mask = torch.tensor(tmp['attention_mask']).unsqueeze(0)
    seg = torch.tensor(tmp['token_type_ids']).unsqueeze(0)
    h_0 = torch.rand(2, input_ids.shape[0], bert_lstm_hyperparameters['lstm_hidden_size'])
    c_0 = torch.rand(2, input_ids.shape[0], bert_lstm_hyperparameters['lstm_hidden_size'])

    output = bert_lstm_model(input_ids, attention_mask, seg, h_0, c_0)
    return output.item()
    


def paragram_embedding_unpooled(phrase):
    return [paragram_model[w] if w in paragram_model.keys() else [0] * hyperparameters['embedding_dim'] for w in word_tokenize(phrase.lower())]




def compute_similarity_siamese(s1, s2):
    input_1 = torch.tensor(paragram_embedding_unpooled(s1)).unsqueeze(0)
    input_2 = torch.tensor(paragram_embedding_unpooled(s2)).unsqueeze(0)

    h_0 = torch.rand(1, input_1.shape[0], siamese_hyperparameters['gru_hidden_size'])

    output = siamese_model(input_1, input_2, h_0)
    return output.item()



def compute_similarity_rf(s1, s2):
    return rf_model.predict(np.array(get_features(s1, s2)).reshape(1, -1))[0]



def compute_similarity_ensemble(s1, s2):
    sim = 10 * compute_similarity_bert_agg_match(s1, s2)
    sim += 3 * compute_similarity_siamese(s1, s2)
    sim += 3 * cosine_sim(s1, s2, embedding="paragram")
    sim += compute_similarity_rf(s1, s2)
    sim /= 17
    return sim



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



### BERT Clssification
saved = torch.load("models/bert_classification.pt")
bert_classification_hyperparameters = saved["hyperparameters"]
bert_classification_model = BERT_classification(bert_hidden_size=bert_classification_hyperparameters['bert_hidden_size'], classifier_hidden_size=bert_classification_hyperparameters['classifier_hidden_size'], dropout=bert_classification_hyperparameters['dropout'])
bert_classification_model.load_state_dict(saved['model'])
bert_classification_model.eval()
bert_classification_tokenizer = bert_classification_model.tokenizer



### BERT AGG MATCH
saved = torch.load("models/bert_agg_match.pt")
bert_lstm_hyperparameters = saved["hyperparameters"]
bert_lstm_model = BERT_AGG_MATCH(bert_hidden_size=bert_lstm_hyperparameters['bert_hidden_size'], lstm_hidden_size=bert_lstm_hyperparameters['lstm_hidden_size'], classifier_hidden_size=bert_lstm_hyperparameters['classifier_hidden_size'], dropout=bert_lstm_hyperparameters['dropout'], pooling=bert_lstm_hyperparameters['pooling'])
bert_lstm_model.load_state_dict(saved['model'])
bert_lstm_model.eval()
bert_lstm_tokenizer = bert_lstm_model.tokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')



saved = torch.load("models/siamese.pt")
siamese_hyperparameters = saved["hyperparameters"]
siamese_model = Siamese(embedding_size=siamese_hyperparameters['embedding_dim'], gru_hidden_size=siamese_hyperparameters['gru_hidden_size'], classifier_hidden_size=siamese_hyperparameters['classifier_hidden_size'], dropout=siamese_hyperparameters['dropout'])
siamese_model.load_state_dict(saved['model'])
siamese_model.eval()




w2vec_model =  Word2Vec.load("embeddings/word2vec_sg_100_5_5.model").wv
glove_model = KeyedVectors.load_word2vec_format("embeddings/glove.txt", binary=False)
with open("embeddings/paragram_embed_ws353.pkl", 'rb') as f:
    paragram_model = pickle.load(f)


with open("models/random_forest.pkl", "rb") as f:
    rf_model = pickle.load(f)["model"]


if __name__ == "__main__":
    s1 = "food delicious"
    s2 = "pizza tasty"

    print(compute_similarity_bert_classification(s1, s2))
    print(compute_similarity_bert_agg_match(s1, s2))
    print(compute_similarity_ensemble(s1, s2))


