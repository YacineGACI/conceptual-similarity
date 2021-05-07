import torch 
import torch.nn as nn
import random, math

from models import BERT_AGG_MATCH

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





def run(input, attention_mask, seg, target, mode='train'):
    if mode == 'train':
        model.train()
        model.zero_grad()
    else:
        model.eval()

    h_0 = torch.rand(2, input.shape[0], lstm_hidden_size).to(device)
    c_0 = torch.rand(2, input.shape[0], lstm_hidden_size).to(device)

    output = model(input, attention_mask, seg, h_0, c_0)
    loss = criterion(output, target.unsqueeze(-1))

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

    learning_rate = 0.000005
    n_epochs = 20
    minibatch_size = 32
    weight_decay = 0.0004
    dropout = 0.3
    train_dev_test_split = (0.6, 0.2)
    print_every = 1

    bert_hidden_size = 768
    lstm_hidden_size = 128
    classifier_hidden_size = 256

    max_padding_length = 20
    pooling = "mean"

    model = BERT_AGG_MATCH(bert_hidden_size=bert_hidden_size, lstm_hidden_size=lstm_hidden_size, classifier_hidden_size=classifier_hidden_size, dropout=dropout, pooling=pooling)
    criterion = nn.BCELoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    print("BERT_AGG_MATCH model loaded successfully")

    tokenizer = model.tokenizer

    x_train = []
    attention_mask_train = []
    seg_train = []


    for i in range(len(phrases)):
        tmp = tokenizer.encode_plus(phrases[i][0], phrases[i][1], add_special_tokens=True, padding='max_length', max_length=max_padding_length)
        x_train.append(tmp['input_ids'])
        attention_mask_train.append(tmp['attention_mask'])
        seg_train.append(tmp['token_type_ids'])



    x_train = torch.tensor(x_train)
    attention_mask_train = torch.tensor(attention_mask_train)
    seg_train = torch.tensor(seg_train)
    y_train = torch.tensor(labels)

    # Splitting the data
    boundary_train = int(num_training_example * train_dev_test_split[0])
    boundary_dev = boundary_train + int(num_training_example * train_dev_test_split[1])

    x_dev = x_train[boundary_train: boundary_dev]
    attention_mask_dev = attention_mask_train[boundary_train: boundary_dev]
    seg_dev = seg_train[boundary_train: boundary_dev]
    y_dev = y_train[boundary_train: boundary_dev]


    x_test = x_train[boundary_dev:]
    attention_mask_test = attention_mask_train[boundary_dev:]
    seg_test = seg_train[boundary_dev:]
    y_test = y_train[boundary_dev:]

    x_train = x_train[:boundary_train]
    attention_mask_train = attention_mask_train[:boundary_train]
    seg_train = seg_train[:boundary_train]
    y_train = y_train[:boundary_train]
    print("Dataset processing done")



    train_loss = 0
    dev_loss = 0
    test_loss = 0

    model.train()
    model.to(device)


    num_training_examples = x_train.shape[0]
    num_batches = math.ceil(num_training_examples / minibatch_size)

    num_dev_examples = x_dev.shape[0]
    num_batches_dev = math.ceil(num_dev_examples / minibatch_size)

    num_test_examples = x_test.shape[0]
    num_batches_test = math.ceil(num_test_examples / minibatch_size)


    hyperparameters = {
            'learning_rate': learning_rate,
            'n_epochs': n_epochs,
            'minibatch_size': minibatch_size,
            'weight_decay': weight_decay,
            'dropout': dropout,
            'train_dev_test_split': train_dev_test_split,
            'bert_hidden_size': bert_hidden_size,
            'lstm_hidden_size': lstm_hidden_size,
            'classifier_hidden_size': classifier_hidden_size,
            'loss': 'BCELoss',
            'optimizer': 'Adam',
            'random_seed': 0,
            'torch_seed': 0,
            'num_examples': len(phrases),
            'max_padding_length': max_padding_length,
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
        for i in range(num_batches):
            boundary = i * minibatch_size
            sentence = x_train[boundary: boundary + minibatch_size].to(device)
            attention_mask = attention_mask_train[boundary: boundary + minibatch_size].to(device)
            seg = seg_train[boundary: boundary + minibatch_size].to(device)
            target = y_train[boundary: boundary + minibatch_size].to(device)

            train_loss += run(sentence, attention_mask, seg, target, mode='train')


        for i in range(num_batches_dev):
            boundary = i * minibatch_size
            sentence = x_dev[boundary: boundary + minibatch_size].to(device)
            attention_mask = attention_mask_dev[boundary: boundary + minibatch_size].to(device)
            seg = seg_dev[boundary: boundary + minibatch_size].to(device)
            target = y_dev[boundary: boundary + minibatch_size].to(device)

            dev_loss += run(sentence, attention_mask, seg, target, mode='dev')


        for i in range(num_batches_test):
            boundary = i * minibatch_size
            sentence = x_test[boundary: boundary + minibatch_size].to(device)
            attention_mask = attention_mask_test[boundary: boundary + minibatch_size].to(device)
            seg = seg_test[boundary: boundary + minibatch_size].to(device)
            target = y_test[boundary: boundary + minibatch_size].to(device)

            test_loss += run(sentence, attention_mask, seg, target, mode='test')


        if dev_loss < min_loss_dev:
            min_loss_dev = dev_loss
            best_epoch_dev = epoch
            hyperparameters['n_epochs'] = epoch + 1

            torch.save({
                'model': model.state_dict(),
                'hyperparameters': hyperparameters
            } , "models/BERT_AGG_MATCH.pt")

        
            
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