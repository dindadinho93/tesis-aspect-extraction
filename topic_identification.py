from glove_model import load__glove_model, get_word_based_glove
from dictionary import ldag7_edukasi_5, ldag7_edukasi_7, ldag7_game_5, ldag7_ecommerce_7, ldag7_ecommerce_5, ldag7_game_7, \
    ldag_edukasi_5, ldag_edukasi_7, ldag_ecommerce_5, ldag_ecommerce_7, ldag_game_5, ldag_game_7, ldat_edukasi_5, \
    ldat_ecommerce_5, ldat_ecommerce_7, ldat_game_7, ldat_game_5, ldat_edukasi_7, lda_ecommerce_5, lda_edukasi_5, \
    lda_game_5, lda_edukasi_7, lda_game_7, lda_ecommerce_7, btm_edukasi_5, btm_ecommerce_5, btm_game_7, btm_game_5, btm_edukasi_7, btm_ecommerce_7
from cosine_similarity import cos_similarity,  calculate_threshold
from statistics import mean, harmonic_mean
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
from heapq import nlargest
import math
import nltk

# # TEXT PREPROCCESSING
def load_file_noun_list():
    lem_sent = []
    f = open('data/nounlist.txt', 'r')
    sent = [word for line in f for word in line.lower().split()]
    tag_text = nltk.pos_tag(sent)
    for w, tag in tag_text:
        if tag == 'NN' or tag == 'NNS':
            word1 = WordNetLemmatizer().lemmatize(w)
            lem_sent.append(word1)
    return lem_sent


# # CONVERT TO GLOVE
a = wn.synsets('everybody')
print(a)
dictionary = load_file_noun_list()
data_glove = 'data/glove.twitter.27B.50d.txt'
mod = load__glove_model(data_glove)

gloveDict_topic = get_word_based_glove(dictionary, mod)


# # METHOD UNTUK PATH SIMILARITY
def wordnet_list(topic_wn, aspect_wn):
    wordnet_topic_dict = {}
    wordnet_aspect_dict = {}
    syn = '.n.01'
    for token1 in topic_wn:
        try:
            vector = wn.synset(token1 + syn)
            wordnet_topic_dict[token1] = vector
        except KeyError:
            vector = wn.synsets(token1)
        except:
            vector = 'empty'
    for token2 in aspect_wn:
        try:
            vector = wn.synset(token2 + syn)
            wordnet_aspect_dict[token2] = vector
        except KeyError:
            vector = wn.synsets(token2)
        except:
            vector = 'empty'
    return wordnet_topic_dict, wordnet_aspect_dict


# # METHOD UNTUK MEMBUAT MATRIX PATH
def wordnet_matrix(topic_wn, aspect_wn):
    wn_list = {}
    for token1, val1 in topic_wn.items():
        wn_list[token1] = {}
        for token2, val2 in aspect_wn.items():
            if token1 != token2:
                wn_list[token1][token2] = wn.path_similarity(val1, val2)
    return wn_list


# # METHOD UNTUK MENDAPATKAN RATA-RATA PATH
def average_wn_row(dictionary, wn_list):
    avg_wn_dict = {}
    for token1 in dictionary.keys():
        avg_wn_dict[token1] = mean(wn_list[token1].values())
    # for token1 in dictionary.keys():
    #     avg_lc_dict[token1] = harmonic_mean(lc_list[token1].values())
    return avg_wn_dict


# # METHOD UNTUK LEACHOCK CHODOROW SIMILARITY
def synonym_matrix(aspect_sn):
    syn = '.n.01'
    sn_list = {}
    list_lch = []
    for token1, val1 in aspect_sn.items():

        sn_list[token1] = {}
        for token2, val2 in aspect_sn.items():

            if token1 != token2:
                sn_list[token1][token2] = wn.lch_similarity(val1, val2)
                list_lch.append(sn_list[token1][token2])
    return sn_list, list_lch


# # METHOD UNTUK MENGHITUNG RATA-RATA HARMONI LEACHOCK CHODOROW
def average_sn_row(dictionary, sn_list):
    avg_sn_dict = {}
    for token1 in dictionary.keys():
        avg_sn_dict[token1] = harmonic_mean(sn_list[token1].values())
    return avg_sn_dict


# # METHOD UNTUK MELAKUKAN TOPIC SELECTION
def cosine_topic_matrix(topic_v, aspect_v):
    csm = {}
    for token1 in topic_v:
        csm[token1] = {}
        for token2 in aspect_v:
            if token1 != token2:
                csm[token1][token2] = cos_similarity(topic_v[token1], aspect_v[token2])
    return csm


# # METHOD UNTUK MENGHITUNG RATA-RATA HARMONI DARI COSINE SIMILARITY TOPIC SELECTION
def average_cosine_row(gloveDict, csm):
    avgDict = {}
    for token1 in gloveDict:
        # avgDict[token1] = math.sqrt(harmonic_mean(csm[token1].values()) * mean(csm[token1].values()))
        avgDict[token1] = harmonic_mean(csm[token1].values())
    return avgDict


# # METHOD UTAMA DARI LATENT TOPIC IDENTIFICATION
data_aspect = btm_game_7[2]
list_topic = []
alpha = 0.4
for i in range(1, len(data_aspect) + 1):
    aspect = data_aspect[i]
    max_keys1 = ''
    max_keys = ''
    print('aspect term', aspect)

    # # MEMANGGIL METHOD PATH DAN LEACOCK CODOROW SIMILARITY
    topic_dict, aspect_dict = wordnet_list(dictionary, aspect)
    wn_topic = wordnet_matrix(topic_dict, aspect_dict)
    avg_wn_topic = average_wn_row(topic_dict, wn_topic)

    sn_topic, list_all = synonym_matrix(aspect_dict)
    avg_sn_topic = average_sn_row(aspect_dict, sn_topic)
    threshold_lti = calculate_threshold(list_all, alpha)

    max_value = max(avg_sn_topic.values())  # maximum value
    for k, v in avg_sn_topic.items():
        if v == max_value:  # getting all keys containing the `maximum`
            max_keys = k


    # # KATA DIAMBIL JIKA NILAI LEBIH DARI THRESHOLD LTI
    token_high_value = []
    high_values = nlargest(len(data_aspect) + 5, avg_wn_topic, key=avg_wn_topic.get)
    for h_value in high_values:
        token_high_value.append(h_value)
    if max_value > threshold_lti:
        token_high_value.append(max_keys)

    print('Rank candidate 1', token_high_value)

    # token_max_value = []
    # max_value = nlargest(5, avg_sn_topic, key=avg_sn_topic.get)
    # for max_v in max_value:
    #     token_max_value.append(max_v)
    # print('Rank Candidate 2', token_max_value)

    # token_high_dict1 = get_word_based_glove(token_max_value, mod)
    # gloveDict1_aspect = get_word_based_glove('woman', mod)
    # csm_topic1 = cosine_topic_matrix(token_high_dict1, gloveDict1_aspect)
    # avg_topic1 = average_cosine_row(token_high_dict1, csm_topic1)

    token_high_dict = get_word_based_glove(token_high_value, mod)
    gloveDict_aspect = get_word_based_glove(aspect, mod)
    csm_topic = cosine_topic_matrix(token_high_dict, gloveDict_aspect)
    avg_topic = average_cosine_row(token_high_dict, csm_topic)

    max_value1 = max(avg_topic.values())  # maximum value
    max1 = max(avg_topic.keys(), key=avg_topic.get)
    for k, v in avg_topic.items():
        if v == max_value1:  # getting all keys containing the `maximum`
            max_keys1 = k
    print(max_keys1, max_value1)
    # token_max_value1 = []
    # max_value1 = nlargest(8, avg_topic1, key=avg_topic1.get)
    # for max_v in max_value1:
    #     token_max_value1.append(max_v)
    # print(token_max_value1)
    list_topic.append(max_keys1)
print(list_topic)
