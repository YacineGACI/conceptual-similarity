import random

def read_training_data(filename):
    phrases = []
    labels = []
    with open(filename, 'r') as f:
        for l in f.readlines():
            s1, s2, sim = l.split(',')
            phrases.append((s1.strip(' '), s2.strip(' ')))
            labels.append(float(sim.strip('\n ')))
    return phrases, labels



def write_training_data(filename, phrases, labels):
    assert len(phrases) == len(labels)
    with open(filename, 'w') as f:
        for i in range(len(phrases)):
            f.write("{}, {}, {}\n".format(phrases[i][0], phrases[i][1], labels[i]))



if __name__ == "__main__":
    phrases, labels = read_training_data("dataset/generated/sim_dataset_no_expansion_51_10000.csv")
    assert len(phrases) == len(labels)
    num_training_example = len(phrases)
    print('Training dataset read')

    new_filename = "dataset/generated/training_data_noisy_0.05.csv"
    noise_ratio = 0.05
    new_labels = []

    for l in labels:
        likelihood = random.random()
        new_label = int(l)
        if likelihood < noise_ratio:
            new_label = int(1) if l == 0 else int(0)
        new_labels.append(new_label)

    write_training_data(new_filename, phrases, new_labels)
        
