import numpy as np
import io
from statistics import mean
from heapq import nlargest
import pandas as pd
import csv
from cosine_similarity import cos_similarity_v2
dataset = 'data/glove.twitter.27B.50d.txt'
model = {}
model1 = {}
file_save_vector = 'glove_vector_'


def load__glove_model(data):
    print("Loading Glove Model")

    with open(data, encoding="utf8") as f:
        content = f.readlines()

    for line in content:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model1[word] = embedding
    print("Done.", len(model1), " words loaded!")
    return model1


def load_fast_text(data):
    print("Loading Fast text Model")

    with open(data, encoding="utf8") as f:
        content = f.readlines()

    for line in content:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
    print("Done.", len(model), " words loaded!")
    return model


def get_word_based_glove(dictionary, mod):
    glove_dict = {}
    for i in range(len(dictionary)):
        word = dictionary[i]
        try:
            vector = mod[dictionary[i]]
            glove_dict[word] = vector
        except KeyError:
            vector = "doesnt exist"

    return glove_dict


# def save_glove_model_csv(word, vector):
#     writer = csv.writer(open('C:\\Users\\dinda\\PycharmProjects\\test\\data\\file_' + file_save_vector + '.csv', 'w'))
#     if vector == "doesnt exist":
#         writer.writerow({word})

# mod = load__glove_model(dataset)
#
#
# # def find_closest_word(embedding):
# #     return sorted(model.keys(), key=lambda word: spatial.distance.cosine(model[word], embedding))
#
