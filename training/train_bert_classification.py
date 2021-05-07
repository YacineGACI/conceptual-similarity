import torch 
import torch.nn as nn
import random, math

from models import BERT_classification

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

    output = model(input, attention_mask, seg)
    loss = criterion(output, target.unsqueeze(-1))

    if mode == 'train':
        loss.backward()
        optimizer.step()

    return loss.item()




if __name__ == "__main__":

    # Define the GPU device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    phrases, labels = read_training_data("dataset/generated/sim_dataset_no_expansion_51_10000.csv")
    assert len(phrases) == len(labels)
    num_training_example = len(phrases)
    print('Training dataset read')

    learning_rate = 0.00005
    n_epochs = 40
    minibatch_size = 32
    weight_decay = 0.0004
    dropout = 0.3
    train_test_split = 0.7
    print_every = 1

    bert_hidden_size = 768
    classifier_hidden_size = 512

    max_padding_length = 30

    model = BERT_classification(bert_hidden_size=bert_hidden_size, classifier_hidden_size=classifier_hidden_size, dropout=dropout)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    print("BERT_classification model loaded successfully")

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

    x_test = x_train[int(num_training_example * train_test_split):]
    attention_mask_test = attention_mask_train[int(num_training_example * train_test_split):]
    seg_test = seg_train[int(num_training_example * train_test_split):]
    y_test = y_train[int(num_training_example * train_test_split):]

    x_train = x_train[:int(num_training_example * train_test_split)]
    attention_mask_train = attention_mask_train[:int(num_training_example * train_test_split)]
    seg_train = seg_train[:int(num_training_example * train_test_split)]
    y_train = y_train[:int(num_training_example * train_test_split)]
    print("Dataset processing done")



    train_loss = 0
    test_loss = 0

    model.train()


    model.to(device)


    num_training_examples = x_train.shape[0]
    num_batches = math.ceil(num_training_examples / minibatch_size)

    num_test_examples = x_test.shape[0]
    num_batches_test = math.ceil(num_test_examples / minibatch_size)


    hyperparameters = {
            'learning_rate': learning_rate,
            'n_epochs': n_epochs,
            'minibatch_size': minibatch_size,
            'weight_decay': weight_decay,
            'dropout': dropout,
            'train_test_split': train_test_split,
            'bert_hidden_size': bert_hidden_size,
            'classifier_hidden_size': classifier_hidden_size,
            'loss': 'BCELoss',
            'optimizer': 'Adam',
            'random_seed': 0,
            'torch_seed': 0,
            'num_examples': len(phrases),
            'max_padding_length': max_padding_length
        }

    for epoch in range(n_epochs):
        for i in range(num_batches):
            boundary = i * minibatch_size
            input = x_train[boundary: boundary + minibatch_size].to(device)
            attention_mask = attention_mask_train[boundary: boundary + minibatch_size].to(device)
            seg = seg_train[boundary: boundary + minibatch_size].to(device)
            target = y_train[boundary: boundary + minibatch_size].to(device)

            train_loss += run(input, attention_mask, seg, target, mode='train')


        for i in range(num_batches_test):
            boundary = i * minibatch_size
            input = x_test[boundary: boundary + minibatch_size].to(device)
            attention_mask = attention_mask_test[boundary: boundary + minibatch_size].to(device)
            seg = seg_test[boundary: boundary + minibatch_size].to(device)
            target = y_test[boundary: boundary + minibatch_size].to(device)

            test_loss += run(input, attention_mask, seg, target, mode='test')

            
        if (epoch + 1) % print_every == 0:
            train_loss /= (num_training_examples * print_every)
            test_loss /= (num_test_examples * print_every)
            print("Training {:.2f}% --> Training Loss = {:.4f}".format(round(((epoch + 1) / n_epochs) * 100, 2), train_loss))
            print("Training {:.2f}% --> Test Loss = {:.4f}".format(round(((epoch + 1) / n_epochs) * 100, 2), test_loss))
            print()
            train_loss = 0
            test_loss = 0

        hyperparameters['n_epochs'] = epoch + 1

        torch.save({
            'model': model.state_dict(),
            'hyperparameters': hyperparameters
        } , "models/bert_classification.pt")

    print("Training Complete")