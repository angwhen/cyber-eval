# -*- coding: utf-8 -*-
"""
Created on Mon Dec 04 19:49:43 2017

@author: Angel
"""

import numpy as np
from math import floor
import pickle
from sklearn import svm,naive_bayes,ensemble
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.utils import shuffle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans

def count_vectorizer(all_sents):
    all_sent_strs = []
    for sentence in all_sents:
        sent_str = ""
        for word in sentence:
            sent_str = sent_str + str(word) + " "
        all_sent_strs.append(sent_str)
        
    count_vectorizer = CountVectorizer()
    counts = count_vectorizer.fit_transform(all_sent_strs)
    
    counts = counts.toarray().tolist()
    
    pca = PCA(n_components = 1000)
    pca.fit(counts)

    pca_vectors = pca.transform(counts)
    
    return pca_vectors
    

all_sents = pickle.load(open("../Preprocessing/grouped_all_sents_no_lemma_lcase.p", "rb"))
multi_targets = pickle.load(open("multi_class_labels.p", "rb"))
all_sents = [x for x in all_sents if not (len(x) == 1 and x[0] == "$eof$")]

vectors = count_vectorizer(all_sents)
x_train, x_test, y_train, y_test = train_test_split(vectors,multi_targets,test_size = 0.4)


