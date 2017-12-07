# -*- coding: utf-8 -*-
"""
Created on Mon Dec 04 21:49:03 2017

@author: Angel
"""

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
from sklearn.metrics import average_precision_score, precision_score,recall_score
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

def skipgram(all_sents):
    import fasttext
    
    fasttext_model = fasttext.skipgram('data.txt','model')
    
    sent_vectors = []
    for sentence in all_sents: 
        curr_words_list = []
        if (len(sentence)==0 ): #why are there even len 0 sentences
            curr_vector = [0]*100
        else:
            for word in sentence:
                word_vector = fasttext_model[str(word)]
                curr_words_list.append(word_vector)
            curr_vector = np.mean(np.array(curr_words_list),axis=0).tolist()
        sent_vectors.append(curr_vector)
   
    return sent_vectors
    
def count_vectorizer(all_sents, full = False):
    all_sent_strs = []
    for sentence in all_sents:
        sent_str = ""
        for word in sentence:
            sent_str = sent_str + str(word) + " "
        all_sent_strs.append(sent_str)
        
    count_vectorizer = CountVectorizer()
    counts = count_vectorizer.fit_transform(all_sent_strs)
    
    counts = counts.toarray().tolist()
    
    if full:
        return counts
    
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

def evaluate_model(model,x_test,y_test):
    #evaluating
    predictions = model.predict(x_test)
       
    acc = accuracy_score(y_test,predictions,normalize=True)
    print "accuracy of model is %f" %acc

    f1 = f1_score(y_test,predictions,average='binary')
    print "bin f1 of model is %f" %f1
    
    precision = precision_score(y_test,predictions,average='binary')
    print "precision of model is %f" %precision
    
    recall = recall_score(y_test,predictions,average='binary')
    print "recall of model is %f" %recall

    #confusion matrix
    cnf_matrix = confusion_matrix(y_test, predictions)
    print cnf_matrix

def test_model(x_train,x_test,y_train,y_test):
    #training
    model = svm.SVC(class_weight = 'balanced')
    
    model.fit(x_train, y_train)
    
    evaluate_model(model,x_test,y_test)
    y_score = model.decision_function(x_test) # results seem to be like predict proba
    plot_precision_recall_curve(y_test,y_score)
    
all_sents = pickle.load(open("../Preprocessing/grouped_all_sents_no_lemma_lcase.p", "rb"))
all_sents = [x for x in all_sents if not (len(x) == 1 and x[0] == "$eof$")]
all_targets = pickle.load(open("../Preprocessing/all_labels.p", "rb"))
all_targets = [x for x in all_targets if x != "$EOF$"]

#vectors = count_vectorizer(all_sents,full=True)
vectors = skipgram(all_sents)
x_train, x_test, y_train, y_test = train_test_split(vectors,all_targets,test_size = 0.4)

print "SVM"
test_model(x_train, x_test, y_train, y_test )


