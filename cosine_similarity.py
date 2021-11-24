import numpy as np
from statistics import mean, stdev
from heapq import nlargest
import pandas as pd
import math


def cos_similarity(w1, w2):
    w1 = np.array(w1)
    w2 = np.array(w2)
    dot = np.dot(w1, w2)
    norm_w1 = np.linalg.norm(w1)
    norm_w2 = np.linalg.norm(w2)
    cosine = dot / (norm_w1 * norm_w2)
    return abs(round(cosine, 3))


def cos_similarity_v2(v1, v2, s):
    sumxx, sumxy, sumyy = 0, 0, 0
    for i in range(len(v1)):
        x = v1[i]; y = v2[i]
        sumxx += x * x * s
        sumxy += x * y * s
        sumyy += y * y * s
    csm = sumxy / math.sqrt(sumxx * sumyy)
    if csm == 1.0:
        return 0
    else:
        return csm


def cosine_matrix(dictionary):
    csm = {}
    list_csm = []
    for token1 in dictionary:
        csm[token1] = {}
        for token2 in dictionary:
            csm[token1][token2] = cos_similarity(dictionary[token1], dictionary[token2])
            list_csm.append(csm[token1][token2])
    return csm, list_csm


def average_cosine_row(gloveDict, csm):
    avgDict = {}
    for token1 in gloveDict:
        avgDict[token1] = mean(csm[token1].values())
    return avgDict


def get_n_high_value_row(avgDict, n_topic):
    token_high_value = []
    high_values = nlargest(n_topic, avgDict, key=avgDict.get)
    for h_value in high_values:
        token_high_value.append(h_value)

    return token_high_value


def get_cluster_value(token_high_value, gloveDict, csm):
    csm_high_value = {}
    for token1 in token_high_value:
        csm_high_value[token1] = {}
        for token2 in gloveDict:
            csm_high_value[token1][token2] = csm[token1][token2]

    return csm_high_value


def calculate_threshold(list_csm, alpha):
    avg_threshold = mean(list_csm)
    std_threshold = stdev(list_csm)
    threshold_cal = avg_threshold + (alpha*std_threshold)
    return threshold_cal


def get_max_value_column(csm_high_value, threshold):
    df_high_value = pd.DataFrame(csm_high_value)
    df_high_value['max'] = df_high_value.max(axis=1)
    df_high_value['idxmax'] = df_high_value.idxmax(axis=1)
    df_cluster = pd.DataFrame(index=df_high_value.index)
    df_cluster['cluster'] = df_high_value['idxmax'].where(df_high_value['max'] > threshold)
    df_cluster.dropna(subset=['cluster'], inplace=True)
    return df_cluster


def get_value_cluster_to_index(clusterDict, token_high_value):
    clusterDict_idx = {}
    for idx, info in clusterDict.items():
        for k_cluster in info:
            clusterDict_idx[idx] = token_high_value.index(info[k_cluster]) + 1
    return clusterDict_idx