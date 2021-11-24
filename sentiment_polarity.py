from preprosesing import pre_processing, save_csv_v2, open_csv_v2, open_csv, open_doc
from glove_model import load__glove_model
from tf_idf_model import tf_idf_sklearn
from cosine_similarity import cos_similarity
from aspect_extraction import feature_selection, my_lda_method
from sklearn.model_selection import train_test_split, ShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
import numpy as np
import random
import pandas as pd
import time


# INITIALISATION VARIABLE
rows = 500
starts = 0
filename = 'game_v5'
n_topic = 2
iteration = 10
threshold = 0.7
y_prediction = []
y_act = []

#
# # #   OPEN CSV FOR TEXT PRE_PROCESSING
# start = time.time()  # start time computation
# df = pd.read_csv("data/steam_reviews.csv", error_bad_lines=False)
# text = df['review'][:5000].dropna()
# df['recommendation'] = np.where(df['recommendation'] == 'Recommended', 1, 0)
# label = df['recommendation'][:5000].dropna()
# print('current time: ', time.clock())
# # TEXT PRE_PROCESSING
# doc = pre_processing(text)
# print('current time1: ', time.clock())
# dataset = save_csv_v2(doc, label, filename)
# end = time.time()  # end time computation
#
# print('\nWaktu PRE-PROCESSING: ', end - start)

# WORD EMBEDDING
text1, text2, label = open_csv_v2(filename, starts, rows)
data_glove = 'data/glove.twitter.27B.50d.txt'
mod = load__glove_model(data_glove)
doc = open_doc(filename, starts, rows)
# text2 = open_csv(filename, starts, rows)


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
    label1 = label_list1
    list_label = []
    list_label1 = []
    list_label2 = []

    for lb1 in label1:
        vector1 = mod[lb1]
        list_label1.append(vector1)
    return list_label1


list_dc = np.array(get_doc_based_glove(text1))
list_lbl = np.array(label)
list_dc1 = tf_idf_sklearn(text2).toarray()

X_train = []
X_test = []
y_train = []
y_test = []
test_size = 0.3
rkf = ShuffleSplit(n_splits=2, test_size=test_size, random_state=20)
for train, test in rkf.split(list_dc1):
    X_train = list_dc1[train]
    X_test = list_dc1[test]
    y_train = list_lbl[train]
    y_test = list_lbl[test]
# Split dataset into training set and test set
# X_train, X_test, y_train, y_test = train_test_split(list_dc, list_lbl, test_size=0.3, random_state=20)

# Create a svm Classifier
clf = svm.SVC(kernel='linear')  # Linear Kernel
# Train the model using the training sets
clf.fit(X_train, y_train)
# Predict the response for test dataset
y_pred = clf.predict(X_test)

# Create a DT Classifier
mdt = DecisionTreeClassifier()
mdt.fit(X_train, y_train)
y_pred1 = mdt.predict(X_test)  # prediksi

# Create a MNB Classifier
mnb = GaussianNB()
mnb.fit(X_train, y_train)
y_pred2 = mnb.predict(X_test)

# Create a MLP Classifier
mlp = MLPClassifier(hidden_layer_sizes=(8, 8, 8), activation='relu', solver='adam', max_iter=500)
mlp.fit(X_train, y_train)
y_pred3 = mlp.predict(X_test)

# print(list_dc)
# print(list_lbl)
print('SVM', classification_report(y_test, y_pred))
print('DT', classification_report(y_test, y_pred1))
print('GNB', classification_report(y_test, y_pred2))
print('MLP', classification_report(y_test, y_pred3))

# Jumlah Label
num_zeros = (list_lbl == 0).sum()
num_ones = (list_lbl == 1).sum()
print('Count 0 ', num_zeros)
print('Count 1 ', num_ones)

fs = feature_selection(doc, mod, n_topic)
my_lda, df_lda = my_lda_method(text2, fs, n_topic, iteration)
df_label = df_lda[['dokumen', 'new topic']].drop_duplicates(subset=['dokumen'])
df_label['new topic'] = np.where(df_label['new topic'] == 2, 1, 0)
y_label = df_label['new topic'].values.tolist()
print(y_label)
a = int(rows*test_size*2)
b = len(y_label)
print('Our Method LDA', classification_report(label[:a:2], y_label[:a:2]))
