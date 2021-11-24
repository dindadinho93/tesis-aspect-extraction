import nltk
import string
import re
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from textblob import TextBlob
from gensim.parsing.preprocessing import STOPWORDS
from nltk.corpus import wordnet as wn
from dictionary import CONTRACTION_MAP, EXPAND_STOP_WORD
import pandas as pd
import numpy as np
import time


def tokenizing(text):
    nltk_tokens = nltk.word_tokenize(text)
    return nltk_tokens


def tokenizing_v2(text):
    tkz = RegexpTokenizer(r'\W+')
    result = tkz.tokenize(text)
    return result


def remove_typo(text):
    symbols = "!./-:,;?"
    for i in symbols:
        text = np.char.replace(text, i, ' ')
    return text


def expand_contractions(text, contraction_mapping=CONTRACTION_MAP):
    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())),
                                      flags=re.IGNORECASE | re.DOTALL)

    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match) \
            if contraction_mapping.get(match) \
            else contraction_mapping.get(match.lower())
        expanded_contraction = first_char + expanded_contraction[1:]
        return expanded_contraction

    expanded_text = contractions_pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text


def remove_punctuation(text):
    text = re.sub(r'[^\x00-\x7f]', r' ', text)  # non ASCII
    text = re.sub(r"\s+[a-zA-Z]\s+", " ", text)    # single char
    text = "".join(c for c in text if not c.isdigit())  # digit
    text = text.strip()
    text = text.lower()  # lowercase
    no_punc = "".join([c for c in text if c not in string.punctuation])     # punctuation
    return no_punc


def remove_stopwords_nltk(text):
    stop_words = stopwords.words('english')
    # stop_words.extend(EXPAND_STOP_WORD)

    words = [w for w in text if w not in stop_words]
    return words


def remove_stopwords_gensim(text):
    stop_words_g = STOPWORDS
    words = [w for w in text if w not in stop_words_g]
    return words


def get_phrase(text):
    words = TextBlob(text)
    word = []
    for w, p in words.pos_tags:
        # FOR ASPECT
        # if p == 'NN' or p == 'NNS':
        # FOR SENTIMENT
        if p == 'JJ' or p == 'JJR' or p == 'JJS' or p == 'RB' or p == 'RBR' or p == 'RBS':
            word.append(w)
            word.append(" ")
    return "".join(word)


def get_noun(text):
    words = []
    text = tokenizing(text)
    tag_text = nltk.pos_tag(text)
    for word, tag in tag_text:
        if tag == 'NN' or tag == 'NNS':
            words.append(word)
            words.append(" ")
    return "".join(words)


def word_lemmatizer(text):
    lemma_word = []
    wordnet_lemmatizer = WordNetLemmatizer()
    for w in text:
        word1 = wordnet_lemmatizer.lemmatize(w, pos="n")
        word2 = wordnet_lemmatizer.lemmatize(word1, pos="s")
        word3 = wordnet_lemmatizer.lemmatize(word2, pos=("r"))
        # POS TAG (a, s, r) For Sentiment
        # POST TAG (N) For Aspect
        lemma_word.append(word1)
        lemma_word.append(" ")
    return "".join(lemma_word)


def spell_correction(text):
    return str(TextBlob(text).correct())


def pre_processing(text):
    text = text.apply(lambda x: expand_contractions(x))
    text = text.apply(lambda x: remove_typo(x))
    text = text.apply(lambda x: remove_punctuation(x))
    # text = text.apply(lambda x: get_phrase(x))
    # text = text.apply(lambda x: get_noun(x))
    text = text.apply(lambda x: tokenizing(x))
    text = text.apply(lambda x: remove_stopwords_nltk(x))
    text = text.apply(lambda x: remove_stopwords_gensim(x))
    lemma_word = text.apply(lambda x: word_lemmatizer(x))
    print('current time: ', time.clock())
    return lemma_word


def save_csv(text, file):
    dict = {'review': text}
    df = pd.DataFrame(dict)
    df.to_csv(r'C:\Users\dinda\PycharmProjects\test\data\file_'+file+'.csv', index=False)
    print('data file_'+file+' telah disimpan..!!!')


def save_csv_v2(text, label, file):
    dict1 = {'review': text, 'label': label}
    df = pd.DataFrame(dict1)
    df.to_csv(r'C:\Users\dinda\PycharmProjects\test\data\file_'+file+'.csv', index=False)
    print('data file_'+file+' telah disimpan..!!!')


def open_csv(filename, starts, rows):
    df1 = pd.read_csv('data/file_' + filename + '.csv', error_bad_lines=False)
    doc = df1['review'][starts:rows].dropna()
    return doc


def open_doc(filename, starts, rows):
    doc = open_csv(filename, starts, rows)
    text1 = doc.apply(lambda x: tokenizing(x))
    return text1


def open_csv_v2(filename, starts, rows):
    df1 = pd.read_csv('data/file_' + filename + '.csv', error_bad_lines=False)
    df1 = df1.dropna(how='any', subset=['review', 'label'])
    df1 = df1[starts:rows]
    data = df1['review'].apply(lambda x: tokenizing(x))
    data1 = df1['review']
    label = df1['label']
    return data, data1, label



