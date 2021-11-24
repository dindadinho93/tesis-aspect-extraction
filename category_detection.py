from preprosesing import open_csv, open_doc, open_csv_v2
from glove_model import load__glove_model
from cosine_similarity import cos_similarity
from dictionary import ecommerce_5, edukasi_5, manual_edukasi_5, manual_ecommerce_5, manual_game_5, lti_ecommerce_5
from aspect_extraction import feature_selection, my_lda_method
from statistics import mean
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

# INISIALISASI
rows = 500
starts = 0
filename = 'game_v3'
y_pred = []
y_act = []
n_topic = 5
iteration = 4

# WORD EMBEDDING
doc = open_doc(filename, starts, rows)
text1 = open_csv(filename, starts, rows)
data_glove = 'data/glove.twitter.27B.50d.txt'
mod = load__glove_model(data_glove)


# METHOD CONVERT WORD TO GLOVE
def get_doc_based_glove(text):
    sent = []
    list_sent = []
    for sentence in text:
        for word in sentence:
            try:
                vector1 = mod[word]
                sent = np.mean([vector1], axis=0)
            except KeyError:
                vector1 = 'doesnt exist'
        list_sent.append(sent)
    return list_sent


# METHOD CONVERT CENTROID TO GLOVE
def get_label_based_glove(label_list1):
    label1 = label_list1[0]
    # label2 = label_list2[0]
    list_label = []
    list_label1 = []
    list_label2 = []

    for lb1 in label1:
        vector1 = mod[lb1]
        list_label1.append(vector1)
    # for lb2 in label2:
    #     vector2 = mod[lb2]
    #     list_label2.append(vector2)
    # list3 = [list(a) for a in zip(list_label1, list_label2)]
    # for i in range(len(list3)):
    #     label = np.mean(list3[i], axis=0)
    #     list_label.append(label)
    return list_label1


# METHOD COSINE SIMILARITY
def cosine_matrix(list_doc, list_label):
    csm = {}
    for i in range(len(list_doc)):
        csm[i] = []
        for j in range(len(list_label)):
            cosine = cos_similarity(list_doc[i], list_label[j])
            csm[i].append(cosine)
    return csm


# METHOD AVERAGE COSINE SIMILARITY
def get_high_cosine_row(list_csm):
    for k, v in list_csm.items():
        max_value = max(v)
        index_value = v.index(max_value) + 1
        list_topic = {k + 2: index_value}
        print(list_topic)
        y_pred.append(index_value)
        y_act.append(index_value)


list_dc = get_doc_based_glove(doc)
list_lbl = get_label_based_glove(manual_game_5)

cm = cosine_matrix(list_dc, list_lbl)

get_high_cosine_row(cm)


# METHOD K-MEANS
def k_means(data, centroid, current):
    csm = {}
    cluster = []
    count_data = {}
    list_centroid = []
    for i in range(len(data)):
        csm[i] = []
        for j in range(len(centroid)):
            cosine = 2 - 2*cos_similarity(data[i], centroid[j])
            csm[i].append(cosine)
    for k, v in csm.items():
        min_value = min(v)
        index_value = v.index(min_value) + 1
        cluster.append(index_value)
    count_data['data'] = data
    count_data['cluster'] = cluster
    df_kmean = pd.DataFrame(count_data)
    # print(df_kmean)
    for i in range(len(list_lbl)):
        new_centroid = np.mean(df_kmean[df_kmean['cluster'] == i+1]['data'])
        list_centroid.append(new_centroid)
    if current != 0:
        # print('proses')
        return k_means(list_dc, list_centroid, current - 1)
    else:
        return cluster


# for i in range(20):
#     result_kmeans = k_means(list_dc, list_lbl, i)
#     f1 = f1_score(y_pred[:450], result_kmeans[:450], average='macro')
#     print(i, f1)
#

text, text2, label_hum = open_csv_v2(filename, starts, rows)
#
# for i in range(20):
#     result_kmeans = k_means(list_dc, list_lbl, i)
#     f1 = f1_score(label_hum[:450], result_kmeans[:450], average='macro')
#     print(i, f1)

# LDA
fs = feature_selection(doc, mod, n_topic)
my_lda, df_lda = my_lda_method(text1, fs, n_topic, iteration)
df_label = my_lda[['dokumen', 'new topic']].drop_duplicates(subset=['dokumen'])
# print(my_lda)
label = df_label['new topic'].values.tolist()
f1lda = f1_score(label_hum[:450], label[:450], average='macro')
print(f1lda)