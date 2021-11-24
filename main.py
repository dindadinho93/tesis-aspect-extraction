from preprosesing import pre_processing, save_csv, open_csv, open_doc
from lda_model import create_lda_gensim, create_lda_sklearn, display_topics, perplexity_lda_sklearn, count_text_word, \
    count_doc_topic, count_word_topic, total_per_topic, total_per_doc, probability_topic, calculate_lda, lda_topic_random
from aspect_extraction import feature_selection, my_lda_method, lda_btm, lda_base
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
    ldag7_game_5, lti_ecommerce_5, lti_ecommerce_7, lti_edukasi_5, lti_edukasi_7, lti_game_5, lti_game_7, lti_ecommerce_5_1000, \
    lti_ecommerce_5_1500, lti_edukasi_5_1000, lti_edukasi_5_1500, lti_game_5_1000, lti_game_5_1500, edukasi_5_1000, \
    edukasi_5_1500, ecommerce_5_1000, ecommerce_5_1500, game_5_1000, game_5_1500, lti_ecommerce_7_1000, lti_ecommerce_7_1500, \
    lti_edukasi_7_1000, lti_edukasi_7_1500, lti_game_7_1000, lti_game_7_1500, ecommerce_7_1000, ecommerce_7_1500, edukasi_7_1000, edukasi_7_1500, game_7_1000, game_7_1500
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
from nltk.stem import WordNetLemmatizer
from nltk import bigrams
from textblob import TextBlob
from itertools import combinations
from wordcloud import WordCloud
import matplotlib.pyplot as plt


# INITIALISATION VARIABLE
rows = 500
starts = 0
filename = 'edukasi_v3'
# FOR WORD2VEC DATASET
filename2 = 'e-commerce_no'
n_topic = 7
iteration = 10
threshold = 0.7
# 0 is glove 0.7, 1 is glove, 2 is LDA-tfidf, 3 is LDA
list_topic = game_7_1000[4]
# lda sort 3 0-500 1-1000 2-1500
lda_sort3 = btm_game_7[1]

# # file_save_vector = 'glove_vector_' + filename
# #
# #
# # #   OPEN CSV FOR TEXT PRE_PROCESSING
# start = time.time()  # start time computation
# df = pd.read_csv("data/coursera_reviews.csv", error_bad_lines=False)
# text = df['Review'][:5000].dropna()
# print('current time: ', time.clock())
# # TEXT PRE_PROCESSING
# doc = pre_processing(text)
# print('current time1: ', time.clock())
# dataset = save_csv(doc, filename)
# end = time.time()  # end time computation
#
# print('\nWaktu PRE-PROCESSING: ', end - start)

#   OPEN CSV AFTER TEXT PRE_PROCESSING
start = time.time()  # start time computation
text1 = open_csv(filename, starts, rows)
# text2 = open_csv(filename2, rows*2)
doc = open_doc(filename, starts, rows)
# df2 = pd2.read_csv("data/file_edukasi.csv", error_bad_lines=False)
# text1 = df2['review'][:rows].dropna()
#
#

#   CONVERT WORD TO GLOVE
data_glove = 'data/glove.twitter.27B.50d.txt'
mod = load__glove_model(data_glove)
#   EVALUASI W2V
# glove_input_file = 'data/glove.6B.100d.txt'
# word2vec_output_file = 'data/glove.6B.100d.txt.word2vec'
# glove2word2vec(glove_input_file, word2vec_output_file)
filename_glove = 'data/glove.6B.100d.txt.word2vec'

# #   CONVERT WORD TO FAST TEXT
# data_fast = 'data/wiki-news-300d-1M.vec'
# mod_fast = load_fast_text(data_fast)

fs = feature_selection(doc, mod, n_topic)
my_lda, lda_sort1 = my_lda_method(text1, fs, n_topic, iteration)
print(my_lda)
#
# lda_sort1 = lda_btm(text1, n_topic)
# print(lda_sort1)
#
# df_lda, lda_sort1 = lda_base(doc, text1, n_topic, iteration)

# file_result = 'result'
# df_lda_t.to_csv(r'C:\Users\dinda\PycharmProjects\test\data\file_'+file_result+'.csv', index=False)

#
# #  EVALUATION MODEL


# prob_ev = {pv: sum(1 for ev in freq.values() if pv in ev) for pv in set(x for y in freq.values() for x in y)}
# print(prob_ev)

# list_pmi = {}
# list_pmi2 = {}
# list_npmi = {}
# list_umass = {}
# # for i, j in enumerate(TRHESHOLD_PARAMETER):
# j = 0.99
# cal_pmi = pmi_matrix(lda_sort1, text1, j)
# cal_pmi2 = pmi2_matrix(lda_sort1, text1, j)
# cal_npmi = npmi_matrix(lda_sort1, text1, j)
# cal_umass = umass_matrix(lda_sort1, text1, j)
# list_pmi[n_topic] = avg_pmi(cal_pmi)
# list_pmi2[n_topic] = avg_pmi2(cal_pmi2)
# list_npmi[n_topic] = avg_npmi(cal_npmi)
# list_umass[n_topic] = avg_umass(cal_umass)

# df_pmi = pd.DataFrame(list_pmi)
# df_pmi2 = pd.DataFrame(list_pmi2)
# df_npmi = pd.DataFrame(list_npmi)
# df_umass = pd.DataFrame(list_umass)
# df_pmi1 = df_pmi.round(3)
# df_pmi21 = df_pmi2.round(3)
# df_npmi1 = df_npmi.round(3)
# df_umass1 = df_umass.round(3)
# df_pmi1['mean'] = df_pmi1[n_topic].mean(axis=0)
# df_pmi21['mean'] = df_pmi21[n_topic].mean(axis=0)
# df_npmi1['mean'] = df_npmi1[n_topic].mean(axis=0)
# df_umass1['mean'] = df_umass1[n_topic].mean(axis=0)
# print(df_pmi1)
# print(df_npmi1)
# print(df_umass1)
# print(df_pmi21)
# file_result = 'result'
# df_umass1.to_csv(r'C:\Users\dinda\PycharmProjects\test\data\file_'+file_result+'.csv', index=False)

# # EVALUASI SEMANTIC
# #
# eval_csm_semantic, total_sem = csm_semantic(lda_sort3, mod, list_topic)
# eval_fast_semantic, total_fast = csm_fast_semantic(lda_sort3, mod_fast, list_topic)
# print(list_topic)
# print('cosine', eval_csm_semantic)
# print('total cosine', total_sem)
# eval_wn_semantic, eval_hpn_semantic, eval_lin_semantic, total_lin, total_hpn, total_wn = wn_semantic(lda_sort3, list_topic)
# print('wordnet', eval_wn_semantic)
# print('path', eval_hpn_semantic)
# print('lin', eval_lin_semantic)
# print('total lin', total_lin)
# print('total path', total_hpn)
# print('total wordnet', total_wn)
#
# eval_w2v_semantic, total_w2v = w2v_semantic(lda_sort3, list_topic, text1, filename_glove)
# print('w2v', eval_w2v_semantic)
# print('total w2v', total_w2v)
# print('fast text', eval_fast_semantic)
# print('total fast text', total_fast)

# #
# #    VISUALIZE DATA
# #
# count_item_edu = count_list_topic(edukasi_5, edukasi_7)
# count_item_eco = count_list_topic(ecommerce_5, ecommerce_7)
# count_item_game = count_list_topic(game_5, game_7)
# print('edukasi', count_item_edu)
# print('ecommerce', count_item_eco)
# print('game', count_item_game)
#
# name_bar = list(count_item_eco.keys())
# values_bar = list(count_item_eco.values())
# plt.barh(range(len(count_item_eco)), values_bar, tick_label=name_bar)
# plt.show()
# wc = WordCloud(background_color='white').generate_from_frequencies(count_item_game)
# plt.figure()
# plt.imshow(wc, interpolation="bilinear")
# plt.axis("off")
# plt.show()

end = time.time()  # end time computation
print('\nWaktu komputasi: ', end - start)

# for k, v in lda_sort3.items():
#     x = " ".join(v)
#     print(x)
#
# tfidf = tf_idf_sklearn(text1)
# tfidf2 = tf_idf_sklearn_v2(text1, False)
# tf_feature_name = tf_idf_sklearn_v2(text1, True)
# print("tf-idf: ", tfidf)
# print(tf_feature_name)
#
# print("tf-idf: ", tfidf2)
# tfidf_corpus = tf_idf_gensim(corpus)
# # print("corpus: ", corpus)
# # print("tfidf: ", tfidf_corpus)
#
# # Build LDA model
# lda_model1 = create_lda_gensim(tfidf_corpus, id2word, 5)
# lda_model = create_lda_sklearn(tfidf, 5, 10)
#
# # Print the Keyword in the 10 topics
# pprint(lda_model1.print_topics())
# display_topics(lda_model, tf_feature_name, 10)
#
# end = time.time()  # end time computation
#
# # Compute Perplexity
# print('\nPerplexity sklearn: ', perplexity_lda_sklearn(tfidf, 6, 50))
# print('\nPerplexity: ', lda_model1.log_perplexity(tfidf_corpus))  # a measure of how good the model is. lower the better
# print('\nWaktu komputasi: ', end - start)
