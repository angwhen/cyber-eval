# -*- coding: utf-8 -*-
"""
Created on Mon Dec 04 19:49:43 2017

@author: Angel
"""
from collections import Counter
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
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

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
    
def plot_precision_recall_curve(y_test,y_score):
    average_precision = average_precision_score(y_test, y_score)
    precision, recall, _ = precision_recall_curve(y_test, y_score)
    plt.step(recall, precision, color='b', alpha=0.2,
             where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2,
                     color='b')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
              average_precision))
    
def get_binary_vers(multi):
    bin_vers = []
    for m in multi:
        if m > 1:
            bin_vers.append(0)
        else:
            bin_vers.append(1)
    return bin_vers

def evaluate_model(model,x_test,y_test):
    #evaluating
    predictions = model.predict(x_test)
    
    bin_predictions = get_binary_vers(predictions)
    bin_y_test = get_binary_vers(y_test)
       
    acc = accuracy_score(bin_y_test,bin_predictions,normalize=True)
    print "one vs rest accuracy of model is %f" %acc

    f1 = f1_score(bin_y_test,bin_predictions,average='binary')
    print "one vs rest bin f1 of model is %f" %f1

    #confusion matrix
    cnf_matrix = confusion_matrix(y_test, predictions)
    print cnf_matrix

def test_model(x_train,y_train,x_test,y_test,model_type):
    #training
    if model_type == "svm":
        model = svm.SVC()
    elif model_type == "nb":
        model = naive_bayes.GaussianNB()
    elif model_type =="rf":
        model = ensemble.RandomForestClassifier(max_depth=50, random_state=0)

    model.fit(x_train, y_train)
    
    evaluate_model(model,x_test,y_test)
    if model_type == "svm":
        y_score = model.decision_function(x_test) # results seem to be like predict proba
        bin_y_score = []
        for score_tup in y_score:
            bin_y_score.append(score_tup[1])
        plot_precision_recall_curve(get_binary_vers(y_test),bin_y_score)
    
all_sents = pickle.load(open("../Preprocessing/grouped_all_sents_no_lemma_lcase.p", "rb"))
all_sents = [x for x in all_sents if not (len(x) == 1 and x[0] == "$eof$")]
multi_targets = pickle.load(open("multi_class_labels_k6.p", "rb"))
print Counter(multi_targets)

vectors = count_vectorizer(all_sents)
x_train, x_test, y_train, y_test = train_test_split(vectors,multi_targets,test_size = 0.4)

print "NB"
test_model(x_train,y_train,x_test,y_test,"nb")
print "RF (50)"
test_model(x_train,y_train,x_test,y_test,"rf")
print "SVM"
test_model(x_train,y_train,x_test,y_test,"svm")
