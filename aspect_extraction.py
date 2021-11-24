from lda_model import create_lda_gensim, create_lda_sklearn, display_topics, perplexity_lda_sklearn, count_text_word, \
    count_doc_topic, count_word_topic, total_per_topic, total_per_doc, probability_topic, calculate_lda, lda_topic_random
from tf_idf_model import tf_idf_sklearn, tf_idf_gensim, tf_idf_sklearn_v2, bag_of_words, tf_idf_lda
from glove_model import load__glove_model, get_word_based_glove, load_fast_text
from cosine_similarity import cos_similarity, cosine_matrix, average_cosine_row, get_n_high_value_row, get_cluster_value, \
    get_max_value_column, get_value_cluster_to_index, calculate_threshold
from evaluation_model import calculate_pmi, get_text_pmi, calculate_npmi, pmi_matrix, avg_pmi, npmi_matrix, avg_npmi, \
    umass_matrix, avg_umass, pmi2_matrix, avg_pmi2, csm_semantic, wn_semantic, w2v_semantic, csm_fast_semantic, count_list_topic
from dictionary import TRHESHOLD_PARAMETER, game_5, game_7, edukasi_5, edukasi_7, ecommerce_5, ecommerce_7, \
    lda_edukasi_5, lda_edukasi_7, lda_ecommerce_5, lda_ecommerce_7, lda_game_5, lda_game_7, ldat_edukasi_5, \
    ldat_edukasi_7, ldat_ecommerce_5, ldat_ecommerce_7, ldat_game_5, ldat_game_7, btm_ecommerce_5, btm_ecommerce_7, \
    btm_edukasi_5, btm_edukasi_7, btm_game_5, btm_game_7, ldag_edukasi_5, ldag_ecommerce_7, ldag_edukasi_7, ldag_game_7, \
    ldag_game_5, ldag_ecommerce_5, ldag7_edukasi_5, ldag7_ecommerce_7, ldag7_edukasi_7, ldag7_ecommerce_5, ldag7_game_7, \
    ldag7_game_5, lti_ecommerce_5, lti_ecommerce_7, lti_edukasi_5, lti_edukasi_7, lti_game_5, lti_game_7
from pprint import pprint
import gensim
from gensim import corpora, models
from gensim.scripts.glove2word2vec import glove2word2vec
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import nltk
import pandas as pd
import numpy as np
import time
import string
from statistics import mean, stdev
from collections import defaultdict, Counter
from csv import DictWriter
from csv import writer
from nltk.stem import WordNetLemmatizer
from nltk import bigrams
from textblob import TextBlob
from itertools import combinations
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from biterm.utility import vec_to_biterms, topic_summuary
from biterm.btm import oBTM


threshold = 0.7


def feature_selection(doc, mod, n_topic):
    dictionary = corpora.Dictionary(doc)
    gloveDict = get_word_based_glove(dictionary, mod)
    # MENCARI COSINE SIMILARITY WITH CSM
    print('# MENCARI COSINE SIMILARITY WITH CSM')
    csm, list_csm = cosine_matrix(gloveDict)
    print(len(dictionary))

    # MENCARI AVERAGE FROM COSINE SIMILARITY EACH COLUMN IN A ROW
    print('# MENCARI AVERAGE FROM COSINE SIMILARITY EACH COLUMN IN A ROW')
    avgDict = average_cosine_row(gloveDict, csm)
    print(avgDict)
    avg_val = avgDict.values()
    avg_max = max(avg_val)
    avg_min = min(avg_val)
    avg_avg = mean(list_csm)
    print('maximum average ', avg_max)
    print('minimum average ', avg_min)

    threshold_cal = calculate_threshold(list_csm, 0.99)
    print('threshold ', threshold_cal)
    # MENCARI N HIGH VALUE EACH ROW
    print('# MENCARI N HIGH VALUE EACH ROW')
    token_high_value = get_n_high_value_row(avgDict, n_topic)
    print('cluster point', token_high_value)

    # MENDAPATKAN TOKEN HIGH VALUE
    print('# MENDAPATKAN TOKEN HIGH VALUE (CLUSTER POINT)')
    csm_high_value = get_cluster_value(token_high_value, gloveDict, csm)
    print(csm_high_value)

    # MENCARI NILAI MAKSIMAL VALUE TIAP COLUMN
    print('# MENCARI NILAI MAKSIMAL VALUE TIAP COLUMN')
    df_cluster = get_max_value_column(csm_high_value, threshold)
    print(df_cluster)
    print('jumlah kata : ', len(df_cluster.index))

    # CONVERT DATA FRAME TO DICTIONARY
    print('# CONVERT DATA FRAME TO DICTIONARY')
    clusterDict = df_cluster.to_dict('index')
    print(clusterDict)

    # CONVERT VALUE CLUSTER TO INDEX
    print('# CONVERT VALUE CLUSTER TO INDEX')
    clusterDict_idx = get_value_cluster_to_index(clusterDict, token_high_value)
    print(clusterDict_idx)
    return clusterDict_idx


def my_lda_method(text1, clusterDict_idx, n_topic, iteration):
    # MENGHITUNG JUMLAH KATA
    print('# MENGHITUNG JUMLAH KATA ')
    count_word, word_in_text = count_text_word(text1, clusterDict_idx)

    jumlah_kata = len(count_word.keys())
    print('jumlah kata : ', jumlah_kata)

    # MENGHITUNG JUMLAH KATA PER TOPIK (WORD TO TOPIK)
    print('# MENGHITUNG JUMLAH KATA PER TOPIK (WORD TO TOPIK)')
    word_lda = count_word_topic(clusterDict_idx, n_topic, word_in_text)
    print(word_lda)

    # MENGHITUNG JUMLAH TOPIK DALAM DOKUMEN (DOC TO TOPIC)
    print('# MENGHITUNG JUMLAH TOPIK DALAM DOKUMEN (DOC TO TOPIC)')
    doc_word_lda = count_doc_topic(text1, clusterDict_idx)
    print(doc_word_lda)

    df_lda, dict_lda = calculate_lda(text1, clusterDict_idx, n_topic, iteration, word_lda, doc_word_lda)
    print('\n', df_lda)
    # print(dict_lda)

    lda_sort = {}
    for k, v in dict_lda.items():
        lda_sort[k] = []
        for i, j in sorted(v.items(), key=lambda x: x[1], reverse=True):
            lda_sort[k].append(i)

    lda_sort1 = {x: yl[:10] for x, yl in lda_sort.items()}
    print(lda_sort1)
    df_lda_sort = pd.DataFrame.from_dict(lda_sort1, orient='index')
    df_lda_t = df_lda_sort.transpose()
    return df_lda_t, df_lda
#   you can use df_lda_t to look list of aspect extraction or you can use df_lda to calculate category detection


def lda_base(doc, text1, n_topic, iteration):
    # LDA- TF-IDF
    # ganti tf-idf dengan lda randoom
    dictionary = corpora.Dictionary(doc)
    # RANDOM TOPIC
    #
    lda_random = lda_topic_random(dictionary, n_topic)
    tf_idf = tf_idf_lda(text1, n_topic)
    print(lda_random)
    feature_lda = lda_random
    count_word, word_in_text = count_text_word(text1, feature_lda)
    # sorted_word = sorted(count_word.items(), key=lambda x: x[1], reverse=True)
    word_lda = count_word_topic(feature_lda, n_topic, word_in_text)
    print(word_lda)
    doc_word_lda = count_doc_topic(text1, feature_lda)
    print(doc_word_lda)
    df_lda, dict_lda = calculate_lda(text1, feature_lda, n_topic, iteration, word_lda, doc_word_lda)
    print('\n', df_lda)
    print(dict_lda)
    lda_sort = {}
    for k, v in dict_lda.items():
        lda_sort[k] = []
        for i, j in sorted(v.items(), key=lambda x: x[1], reverse=True):
            lda_sort[k].append(i)

    lda_sort1 = {x: yl[:10] for x, yl in lda_sort.items()}
    print(lda_sort1)
    df_lda_sort = pd.DataFrame.from_dict(lda_sort1, orient='index')
    df_lda_t = df_lda_sort.transpose()
    return df_lda_t, lda_sort1


def lda_btm(text1, n_topics):
    # BTM  (BITERM TOPIC MODEL)
    #
    vec = CountVectorizer()
    x = vec.fit_transform(text1).toarray()

    vocab = np.array(vec.get_feature_names())
    biterms = vec_to_biterms(x)
    btm = oBTM(num_topics=n_topics, V=vocab)

    print("\n\n Train Online BTM ..")
    for i in range(0, len(biterms), 100): # prozess chunk of 200 texts
        biterms_chunk = biterms[i:i + 100]
        btm.fit(biterms_chunk, iterations=10)
    topics = btm.transform(biterms)

    print("\n\n Topic coherence ..")
    btm = topic_summuary(btm.phi_wz.T, x, vocab, 10)
    return btm