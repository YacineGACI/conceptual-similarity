import torch 
import torch.nn as nn
import random, math, pickle
import numpy as np
from nltk.tokenize import word_tokenize

from models import Siamese

torch.manual_seed(0)
random.seed(0)
torch.autograd.set_detect_anomaly(True)


def read_training_data(filename):
    phrases = []
    labels = []
    with open(filename, 'r') as f:
        for l in f.readlines():
            s1, s2, sim = l.split(',')
            phrases.append((s1.strip(' '), s2.strip(' ')))
            labels.append(float(sim.strip('\n ')))
    return phrases, labels




def paragram_embedding(phrase):
    return [paragram_model[w] if w in paragram_model.keys() else [0] * embedding_dim for w in word_tokenize(phrase.lower())]



def run(s1, s2, target, mode='train'):
    if mode == 'train':
        model.train()
        model.zero_grad()
    else:
        model.eval()

    input_1 = torch.tensor(paragram_embedding(s1)).unsqueeze(0).to(device)
    input_2 = torch.tensor(paragram_embedding(s2)).unsqueeze(0).to(device)

    h_0 = torch.rand(1, input_1.shape[0], gru_hidden_size).to(device)

    output = model(input_1, input_2, h_0)
    loss = criterion(output, torch.tensor(target))

    if mode == 'train':
        loss.backward()
        optimizer.step()

    return loss.item()






if __name__ == "__main__":

    # Define the GPU device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    print(device)

    phrases, labels = read_training_data("dataset/generated/sim_dataset_no_expansion_51_10000.csv")
    # phrases, labels = read_training_data("data/training_data_noisy_0.05.csv")
    assert len(phrases) == len(labels)
    num_training_example = len(phrases)
    print('Training dataset read')

    learning_rate = 0.00005
    n_epochs = 20
    minibatch_size = 1
    weight_decay = 0.0004
    dropout = 0.3
    train_dev_test_split = (0.6, 0.2)
    print_every = 1

    embedding_dim = 300
    gru_hidden_size = 256
    classifier_hidden_size = 256

    pooling = "mean"

    model = Siamese(embedding_size=embedding_dim, gru_hidden_size=gru_hidden_size, classifier_hidden_size=classifier_hidden_size, dropout=dropout)
    criterion = nn.BCELoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    print("Siamese model loaded successfully")


    with open("embeddings/paragram_embed_ws353.pkl", 'rb') as f:
        paragram_model = pickle.load(f)



    # Splitting the data
    boundary_train = int(num_training_example * train_dev_test_split[0])
    boundary_dev = boundary_train + int(num_training_example * train_dev_test_split[1])

    x_dev = phrases[boundary_train: boundary_dev]
    y_dev = labels[boundary_train: boundary_dev]


    x_test = phrases[boundary_dev:]
    y_test = labels[boundary_dev:]

    x_train = phrases[:boundary_train]
    y_train = labels[:boundary_train]
    print("Dataset processing done")



    train_loss = 0
    dev_loss = 0
    test_loss = 0

    model.train()
    model.to(device)


    num_training_examples = len(x_train)
    num_dev_examples = len(x_dev)
    num_test_examples = len(x_test)


    hyperparameters = {
            'learning_rate': learning_rate,
            'n_epochs': n_epochs,
            'minibatch_size': minibatch_size,
            'weight_decay': weight_decay,
            'dropout': dropout,
            'train_dev_test_split': train_dev_test_split,
            'embedding_dim': embedding_dim,
            'gru_hidden_size': gru_hidden_size,
            'classifier_hidden_size': classifier_hidden_size,
            'loss': 'BCELoss',
            'optimizer': 'Adam',
            'random_seed': 0,
            'torch_seed': 0,
            'num_examples': len(phrases),
            'pooling': pooling
        }

    # To save the best performoing epoch on the dev set
    best_epoch_train = 0
    best_epoch_dev = 0
    best_epoch_test = 0
    min_loss_train = float('inf')
    min_loss_dev = float('inf')
    min_loss_test = float('inf')

    for epoch in range(n_epochs):
        for i in range(len(x_train)):
            train_loss += run(x_train[i][0], x_train[i][1], y_train[i], mode='train')

        for i in range(len(x_dev)):
            dev_loss += run(x_dev[i][0], x_dev[i][1], y_dev[i], mode='dev')

        for i in range(len(x_test)):
            test_loss += run(x_test[i][0], x_test[i][1], y_test[i], mode='test')


        if dev_loss < min_loss_dev:
            min_loss_dev = dev_loss
            best_epoch_dev = epoch
            hyperparameters['n_epochs'] = epoch + 1

            torch.save({
                'model': model.state_dict(),
                'hyperparameters': hyperparameters
            } , "models/siamese.pt")

        
            
        if (epoch + 1) % print_every == 0:
            train_loss /= (num_training_examples * print_every)
            dev_loss /= (num_dev_examples * print_every)
            test_loss /= (num_test_examples * print_every)
            print("Training {:.2f}% --> Training Loss = {:.4f}".format(round(((epoch + 1) / n_epochs) * 100, 2), train_loss))
            print("Training {:.2f}% --> Dev Loss = {:.4f}".format(round(((epoch + 1) / n_epochs) * 100, 2), dev_loss))
            print("Training {:.2f}% --> Test Loss = {:.4f}".format(round(((epoch + 1) / n_epochs) * 100, 2), test_loss))
            print("This epoch: {}  |  Train: {}  |  Dev: {}  |  Test: {}".format(epoch, best_epoch_train, best_epoch_dev, best_epoch_test))
            print()
            train_loss = 0
            dev_loss = 0
            test_loss = 0


    print("Training Complete")