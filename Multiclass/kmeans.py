# -*- coding: utf-8 -*-
"""
Created on Mon Dec 04 19:09:24 2017

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
all_targets = pickle.load(open("../Preprocessing/all_labels.p", "rb"))

#remove all EOF markers from sent vectors and from targets
all_sents = [x for x in all_sents if not (len(x) == 1 and x[0] == "$eof$")]
all_targets = [x for x in all_targets if x != "$EOF$"]

#negative sents
negative_indices = []
for i in xrange(0,len(all_targets)):
    if all_targets[i] == 0:
        negative_indices.append(i)
pickle.dump(negative_indices,open("negative_indices.p","wb"))

negative_vectors = count_vectorizer(np.array(all_sents).take(negative_indices))
#negative_vectors = vectors.take(negative_indices)

kmeans = KMeans(n_clusters=4, random_state=0).fit(negative_vectors)

multi_class_labels = []
negative_class_index = 0
for i in xrange(0,len(all_targets)):
    if all_targets[i] == 0:
        multi_class_labels.append(kmeans.labels_[negative_class_index] +2)
        negative_class_index = negative_class_index +1
    else:
        multi_class_labels.append(1)

print len(multi_class_labels)
print len(all_targets)

pickle.dump(multi_class_labels,open("multi_class_labels","wb"))





