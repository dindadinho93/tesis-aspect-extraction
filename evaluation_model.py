import numpy as np
import math
import warnings
from gensim import models
from gensim.models import KeyedVectors
from statistics import mean
from collections import Counter
from itertools import combinations
from cosine_similarity import cos_similarity
from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet_ic
warnings.simplefilter("ignore", DeprecationWarning)


def get_text_pmi(text1):
    count_doc = []
    freq = {}
    freq1 = {}
    row = 0
    for rows in text1:
        count_doc.append(rows)
        row = len(count_doc)
        words = rows.split()
        freq[row] = []
        for word in words:
            freq[row].append(word)
    return freq, row


def calculate_w_pmi(w, text1):
    freq, row = get_text_pmi(text1)
    result = Counter()
    for vlist in freq.values():
        for i in set(vlist):
            result[i] += 1
    dw = (result[w])
    pw = dw / row
    return dw, pw


def calculate_ww_pmi(w1, w2, text1):
    dw1, pw1 = calculate_w_pmi(w1, text1)
    dw2, pw2 = calculate_w_pmi(w2, text1)
    return pw1 * pw2


def calculate_pmi_numerator(w1, w2, text1):
    freq, row = get_text_pmi(text1)
    cxy = Counter()
    for vlist in freq.values():
        for x, y in map(sorted, combinations(vlist, 2)):
            cxy[(x, y)] += 1
    pw12 = cxy[(w1, w2)]/row
    return pw12


def calculate_pmi(w1, w2, text1, n):
    ww = calculate_ww_pmi(w1, w2, text1)
    w12 = calculate_pmi_numerator(w1, w2, text1)
    pmi = np.log((w12 + n)/ww)
    return pmi


def calculate_pmi2(w1, w2, text1, n):
    pmi_numerator = calculate_pmi_numerator(w1, w2, text1) + n
    pmi = calculate_ww_pmi(w1, w2, text1) + n
    pmi2 = np.power(pmi_numerator, 2)/pmi
    return np.log(pmi2)


def calculate_npmi(w1, w2, text1, n):
    # pmi_numerator = calculate_pmi_numerator(w1, w2, text1)
    # pmi = calculate_ww_pmi(w1, w2, text1)
    # npmi = np.log(np.power(pmi_numerator, 2)/pmi + n)
    pmi = calculate_pmi(w1, w2, text1, n)
    pmi_numerator = calculate_pmi_numerator(w1, w2, text1)
    npmi = pmi/-np.log(pmi_numerator)
    return npmi


def calculate_umass(w1, w2, text1, n):
    dw, pw = calculate_w_pmi(w1, text1)
    ww = calculate_pmi_numerator(w1, w2, text1)
    umass = (ww + n)/dw
    return umass


def pmi_matrix(df_lda, text1, n):
    _pmi = 0
    list_pmi = {}
    for k, vlist in df_lda.items():
        list_pmi[k] = []
        for word1 in vlist:
            for word2 in vlist:
                if k and word1 != word2:
                    _pmi = calculate_pmi(word1, word2, text1, n)
                    list_pmi[k].append(_pmi)
    return list_pmi


def avg_pmi(list_pmi):
    list_avg_pmi = {}
    for k, v in list_pmi.items():
        list_avg_pmi[k] = mean(v)
    return list_avg_pmi


def pmi2_matrix(df_lda, text1, n):
    _pmi2 = 0
    list_pmi2 = {}
    for k, vlist in df_lda.items():
        list_pmi2[k] = []
        for word1 in vlist:
            for word2 in vlist:
                if k and word1 != word2:
                    _pmi2 = calculate_pmi2(word1, word2, text1, n)
                    list_pmi2[k].append(_pmi2)
    return list_pmi2


def avg_pmi2(list_pmi2):
    list_avg_pmi2 = {}
    for k, v in list_pmi2.items():
        list_avg_pmi2[k] = mean(v)
    return list_avg_pmi2


def npmi_matrix(df_lda, text1, n):
    _npmi = 0
    list_npmi = {}
    for k, vlist in df_lda.items():
        list_npmi[k] = []
        for word1 in vlist:
            for word2 in vlist:
                if k and word1 != word2:
                    _npmi = calculate_npmi(word1, word2, text1, n)
                    list_npmi[k].append(_npmi)
    return list_npmi


def avg_npmi(list_npmi):
    list_avg_npmi = {}
    for k, v in list_npmi.items():
        list_avg_npmi[k] = mean(v)
    return list_avg_npmi


def umass_matrix(df_lda, text1, n):
    _umass = 0
    list_umass = {}
    for k, vlist in df_lda.items():
        list_umass[k] = []
        for word1 in vlist:
            for word2 in vlist:
                if k and word1 != word2:
                    _umass = calculate_umass(word1, word2, text1, n)
                    list_umass[k].append(_umass)
    return list_umass


def avg_umass(list_umass):
    list_avg_umass = {}
    for k, v in list_umass.items():
        list_avg_umass[k] = mean(v)
    return list_avg_umass


#
# SEMANTIC EVALUATION

def csm_semantic(lda_sort, mod, list_topic):
    list_sem_word = {}

    for k_list, v_list in lda_sort.items():
        list_sem_word[k_list] = {}
        for index in range(len(v_list)):
            word = v_list[index]
            try:
                vector = mod[word]
            except KeyError:
                vector = mod['thing']
            list_sem_word[k_list][word] = vector
    list_sem_topic = {}
    for topic_index in range(len(list_topic)):
        topic = topic_index + 1
        word = list_topic[topic_index]
        vector = mod[word]
        list_sem_topic[topic] = vector
    list_csm_sem = {}
    dict_csm_sem = {}
    for k_sem_topic, v_sem_topic in list_sem_topic.items():
        dict_csm_sem[k_sem_topic] = {}
        list_csm_sem[k_sem_topic] = []
        for k_sem_word, v_sem_word in list_sem_word.items():
            for k_word, v_word in v_sem_word.items():
                if k_sem_word == k_sem_topic:
                    dict_csm_sem[k_sem_topic][k_word] = cos_similarity(v_sem_topic, v_word)
                    list_csm_sem[k_sem_topic].append(dict_csm_sem[k_sem_topic][k_word])
    list_avg_sem = {}
    total_sem = []
    for k_avg_sem, v_avg_sem in list_csm_sem.items():
        list_avg_sem[k_avg_sem] = mean(v_avg_sem)
    for k_tot_sem, v_tot_lin in list_avg_sem.items():
        total_sem.append(v_tot_lin)
    tot_avg_sem = mean(total_sem)
    return list_avg_sem, tot_avg_sem


def wn_semantic(lda_sort, list_topic):
    syn = '.n.01'
    semcor_ic = wordnet_ic.ic('ic-semcor.dat')
    list_wn_sem = {}
    list_hpn_sem = {}
    list_lin_sem = {}
    for topic_index in range(len(list_topic)):
        topic = topic_index + 1
        word1 = list_topic[topic_index]
        list_wn_sem[topic] = []
        list_hpn_sem[topic] = []
        list_lin_sem[topic] = []
        for k_list, v_list in lda_sort.items():
            for index in range(len(v_list)):
                word2 = v_list[index]
                if k_list == topic:
                    w1 = wn.synset(word1 + syn)
                    try:
                        w2 = wn.synset(word2 + syn)
                    except:
                        w2 = wn.synset('thing.n.01')
                    wnt = wn.wup_similarity(w1, w2)
                    hpn = wn.path_similarity(w1, w2)

                    lin = wn.lin_similarity(w1, w2, semcor_ic)
                    list_wn_sem[topic].append(wnt)
                    list_hpn_sem[topic].append(hpn)
                    list_lin_sem[topic].append(lin)

    list_avg_wn = {}
    list_avg_hpn = {}
    list_avg_lin = {}
    total_lin = []
    total_hpn = []
    total_wn = []
    for k_avg_wn, v_avg_wn in list_wn_sem.items():
        list_avg_wn[k_avg_wn] = mean(v_avg_wn)
    for k_avg_hpn, v_avg_hpn in list_hpn_sem.items():
        list_avg_hpn[k_avg_hpn] = mean(v_avg_hpn)
    for k_avg_lin, v_avg_lin in list_lin_sem.items():
        list_avg_lin[k_avg_lin] = mean(v_avg_lin)
    for k_tot_lin, v_tot_lin in list_avg_lin.items():
        total_lin.append(v_tot_lin)
    tot_avg_lin = mean(total_lin)
    for k_tot_hpn, v_tot_hpn in list_avg_hpn.items():
        total_hpn.append(v_tot_hpn)
    tot_avg_hpn = mean(total_hpn)
    for k_tot_wn, v_tot_wn in list_avg_wn.items():
        total_wn.append(v_tot_wn)
    tot_avg_wn = mean(total_wn)
    return list_avg_wn, list_avg_hpn, list_avg_lin, tot_avg_lin, tot_avg_hpn, tot_avg_wn


def w2v_semantic(lda_sort, list_topic, text1, filename):
    data1 = []
    list_w2v_sem = {}
    for rows in text1:
        words = rows.split()
        data1.append(words)
        data1.append(list_topic)
    word2vec = models.Word2Vec(data1, min_count=2, size=150, window=10, workers=10, iter=10)
    word2vec1 = KeyedVectors.load_word2vec_format(filename, binary=False)
    for topic_index in range(len(list_topic)):
        topic = topic_index + 1
        word1 = list_topic[topic_index]
        list_w2v_sem[topic] = []
        for k_list, v_list in lda_sort.items():
            for index in range(len(v_list)):
                word2 = v_list[index]
                if k_list == topic:
                    try:
                        w2v = word2vec1.similarity(word1, word2)
                    except:
                        w2v = word2vec1.similarity(word1, 'thing')
                    list_w2v_sem[topic].append(w2v)
    print(list_w2v_sem)
    list_avg_w2v = {}
    total_w2v = []
    for k_avg_w2v, v_avg_w2v in list_w2v_sem.items():
        list_avg_w2v[k_avg_w2v] = mean(v_avg_w2v)
    for k_tot_w2v, v_tot_w2v in list_avg_w2v.items():
        total_w2v.append(v_tot_w2v)
    tot_avg_w2v = mean(total_w2v)
    return list_avg_w2v, tot_avg_w2v


def csm_fast_semantic(lda_sort, mod, list_topic):
    list_sem_word = {}

    for k_list, v_list in lda_sort.items():
        list_sem_word[k_list] = {}
        for index in range(len(v_list)):
            word = v_list[index]
            try:
                vector = mod[word]
            except KeyError:
                vector = mod['thing']
            list_sem_word[k_list][word] = vector
    list_sem_topic = {}
    for topic_index in range(len(list_topic)):
        topic = topic_index + 1
        word = list_topic[topic_index]
        vector = mod[word]
        list_sem_topic[topic] = vector
    list_csm_sem = {}
    dict_csm_sem = {}
    for k_sem_topic, v_sem_topic in list_sem_topic.items():
        dict_csm_sem[k_sem_topic] = {}
        list_csm_sem[k_sem_topic] = []
        for k_sem_word, v_sem_word in list_sem_word.items():
            for k_word, v_word in v_sem_word.items():
                if k_sem_word == k_sem_topic:
                    dict_csm_sem[k_sem_topic][k_word] = cos_similarity(v_sem_topic, v_word)
                    list_csm_sem[k_sem_topic].append(dict_csm_sem[k_sem_topic][k_word])
    list_avg_sem = {}
    total = []

    for k_avg_sem, v_avg_sem in list_csm_sem.items():
        list_avg_sem[k_avg_sem] = mean(v_avg_sem)
    for k_tot_fast, v_tot_fast in list_avg_sem.items():
        total.append(v_tot_fast)
    avg_fast = mean(total)
    return list_avg_sem, avg_fast


def count_list_topic(data1, data2):
    count_item = {}
    for items in data1:
        for item in items:
            count_item[item] = count_item.get(item, 0) + 1
    for items in data2:
        for item in items:
            count_item[item] = count_item.get(item, 0) + 1
    return count_item