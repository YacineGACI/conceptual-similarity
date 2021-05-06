from nltk.corpus import wordnet as wn
import itertools, random, math
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import numpy as np


def read_seed_words(filename):
    concepts = {}
    with open(filename, 'r') as f:
        text = f.read()
        atts = text.strip(" \n\t\r").split('\n\n')
        for a in atts:
            lines = a.split('\n')
            concepts[lines[0]] = [[w.strip(' ') for w in l.split(',')] for l in lines[1:]]
    return concepts



def keep_synset(syn, name, pos):
    if syn._pos != pos or syn._name.split('.')[0] != name:
        return False
    return True



def expand_seeds(seeds, capacity):
    '''
        Given lists of seed words, expand them with synonymous or related words using WordNet
        @seeds: dict, containing for each concept, a set of aspect seeds, and an arbitrary number of opinion seeds
        @capacity: the maximum number of expanded words for every seed word
    '''
    for k in seeds.keys():
        for i, l in enumerate(seeds[k]):
            synonyms = []
            for w in l:
                for syn in wn.synsets(w)[:capacity]:
                    if keep_synset(syn, w, 'n' if i == 0 else 's'):
                        for lemma in syn.lemmas():
                            synonyms.append(lemma.name())
            for s in synonyms:
                if s.lower() not in list(itertools.chain.from_iterable(seeds[k])): # itertools to remove the same word in many lists of the same category, in order not to confuse the learning algorithm
                    seeds[k][i].append(s.lower().replace('_', ' '))
    return seeds



# def create_dataset(filename, seeds, size=100000, min_prob_pos=0.5):
#     # threshold = 7.747541239448234 # ==> 75%
#     threshold = 6.912093323226184 # ==> 51%
#     with open(filename, "w") as f:
#         # Step 1 ==> Construct all possible pairs of aspect and opinion terms
#         phrases = {} # Contains all possible combinations of aspect and opinion terms
#         for k in seeds.keys():
#             tmp_per_attribute = []
#             aspects= seeds[k][0]
#             opinions = seeds[k][1:]
#             for opinion_list in opinions:
#                 # tmp_per_opinion = [x[0] + " " + x[1] for x in itertools.product(aspects, opinion_list)]
#                 # tmp_per_opinion = [(x[0] + " " + x[1], lm_loss(x[0], x[1])) for x in itertools.product(aspects, opinion_list)]
#                 tmp_per_opinion = [x[0] + " " + x[1] for x in itertools.product(aspects, opinion_list) if lm_loss(x[0], x[1]) < threshold]
#                 tmp_per_attribute.append(tmp_per_opinion)
#             phrases[k] = tmp_per_attribute
#             print(k)

#         # Step 2 ==> Randomly sample pairs of phrases
#         catagories = list(seeds.keys())
#         num_pos = 0
#         num_neg = 0
#         for _ in range(size):
#             roulette = random.random() # If roulette == 1 ==> Force a positive example
#                                        # If roulette == 0 ==> Randomly take either a pos or neg example
#             if roulette < min_prob_pos:
#                 cat = random.randint(0, len(catagories) - 1)
#                 op = random.randint(0, len(phrases[catagories[cat]]) - 1)
#                 ph1 = random.randint(0, len(phrases[catagories[cat]][op]) - 1)
#                 ph2 = random.randint(0, len(phrases[catagories[cat]][op]) - 1)

#                 num_pos += 1
#                 f.write("{}, {}, {}\n".format(phrases[catagories[cat]][op][ph1], phrases[catagories[cat]][op][ph2], 1))
#             else:
#                 cat1 = random.randint(0, len(catagories) - 1)
#                 op1 = random.randint(0, len(phrases[catagories[cat1]]) - 1)
#                 ph1 = random.randint(0, len(phrases[catagories[cat1]][op1]) - 1)

#                 cat2 = random.randint(0, len(catagories) - 1)
#                 op2 = random.randint(0, len(phrases[catagories[cat2]]) - 1)
#                 ph2 = random.randint(0, len(phrases[catagories[cat2]][op2]) - 1)

#                 sim = 1 if cat1 == cat2 and op1 == op2 else 0
#                 if sim == 1:
#                     num_pos += 1
#                 else:
#                     num_neg += 1
#                 f.write("{}, {}, {}\n".format(phrases[catagories[cat1]][op1][ph1], phrases[catagories[cat2]][op2][ph2], sim))
#         print("POS ==> ", num_pos)
#         print("NEG ==> ", num_neg)
#         print("% POS ==>", math.floor((num_pos / size) * 100), "%")



def create_dataset(seeds, output_filename, size=100000, min_ratio_pos=0.5, filtering_threshold=6.912093323226184):
    with open(output_filename, "w") as f:
        concepts = list(seeds.keys())

        # Step 1 ==> Construct all possible subjective tags (pairs of aspect and opinion terms)
        subjective_tags = {}
        for c in concepts:
            aspects = seeds[c][0]       # List of aspect terms
            opinions = seeds[c][1:]     # List of lists of opinion terms

            meaningful_tags = [] # Will store all possible aspect x opinion combinations that make sense (list of lists: each list corresponds to an opinion set)
            for opinion_list in opinions:
                meaningful_tags.append([x[0] + " " + x[1] for x in itertools.product(aspects, opinion_list) if lm_loss(x[0], x[1]) < filtering_threshold])
            
            subjective_tags[c] = meaningful_tags
            print("Subjective tags for concept " + c + " are constructed")

        # Step 2 ==> Randomly sample pairs of subjective tags
        num_positive_pairs = 0
        num_negative_pairs = 0
        for _ in range(size):
            roulette = random.random() # If roulette == 1 ==> Force a positive example
                                       # If roulette == 0 ==> Randomly take either a pos or neg example
            if roulette < min_ratio_pos:
                # Sample a concept
                c = random.choice(concepts)

                # Sample an opinion set from that concept
                op = random.choice(subjective_tags[c])

                # Sample two subjective tags from that opnion list
                t1 = random.choice(op)
                t2 = random.choice(op)

                num_positive_pairs += 1
                f.write("{}, {}, {}\n".format(t1, t2, 1))

            
            else:
                c1 = random.choice(concepts)
                op1_index = random.randint(0, len(subjective_tags[c1]) - 1)
                t1 = random.choice(subjective_tags[c1][op1_index])

                c2 = random.choice(concepts)
                op2_index = random.randint(0, len(subjective_tags[c2]) - 1)
                t2 = random.choice(subjective_tags[c2][op2_index])

                sim = 1 if c1 == c2 and op1_index == op2_index else 0
                if sim == 1:
                    num_positive_pairs += 1
                else:
                    num_negative_pairs += 1
                f.write("{}, {}, {}\n".format(t1, t2, sim))

        print("POS ==> ", num_positive_pairs)
        print("NEG ==> ", num_negative_pairs)
        print("% POS ==>", math.floor((num_positive_pairs / size) * 100), "%")




def lm_loss(aspect, opinion):
    sentence = "the " + aspect + " is " + opinion
    input_ids = torch.tensor(tokenizer.encode(sentence, add_special_tokens=True)).unsqueeze(0)  # Batch size 1
    outputs = model(input_ids, labels=input_ids)
    loss = outputs[0]
    return loss.item()



if __name__ == "__main__":
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    print("GPT2 model loaded")

    seeds = read_seed_words("seeds.txt")
    seeds = expand_seeds(seeds, 3)
    create_dataset(seeds, "generated/training_data.csv", min_ratio_pos = 0.35)

    # See if I can limit the expansion of seed words into synonyms