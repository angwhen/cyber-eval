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
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import os
import math
from scipy import spatial
import random
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr
import scipy.stats

class Node(object):
    def __init__(self,value,next_node=None):
        self.value = value
        self.reference = next_node

class LinkedList(object):
    def __init__(self,head_node):
        self.head = head_node
    def get_head(self):
        return self.head
    def get_next(self, curr_node):
        return curr_node.next_node

def is_eof(l):
    if (len(l) == 1 and l[0] == "$eof$"):
        return True
    else:
        False
        
def get_unbalanced_train(counts,all_targets):
    counts,all_targets = shuffle(counts,all_targets,random_state = 0)
    unlab_list = []
    lab_list = []
    for sent_vect,label in zip(counts,all_targets):
        if label == 0:
            unlab_list.append(sent_vect)
        else:
            lab_list.append(sent_vect)

    counts_arr = unlab_list + lab_list
    all_targets = [0]*len(unlab_list) + [1]*len(lab_list)
    print type(counts_arr)
    print type(all_targets)

    x_train, x_test, y_train, y_test = train_test_split(counts_arr,all_targets,test_size = 0.4)
    return np.asarray(x_train),np.asarray(y_train),np.asarray(x_test),np.asarray(y_test)

def get_balanced_train(counts,all_targets):
    counts,all_targets = shuffle(counts,all_targets,random_state = 0)
    unlab_list = []
    lab_list = []
    for sent_vect,label in zip(counts,all_targets):
        if label == 0:
            unlab_list.append(sent_vect)
        else:
            lab_list.append(sent_vect)

    counts_arr = unlab_list + lab_list
    all_targets = [0]*len(unlab_list) + [1]*len(lab_list)
    
    split_num = int(floor(len(counts_arr)*0.25))
    x_train = counts_arr[:split_num]+counts_arr[-split_num:]
    y_train  =  all_targets[:split_num]+all_targets[-split_num:]
    x_test = counts_arr[split_num:-split_num]
    y_test  =  all_targets[split_num:-split_num]
    return np.asarray(x_train),np.asarray(y_train),np.asarray(x_test),np.asarray(y_test)


def evaluate_model(model,x_test,y_test):
    #evaluating
    predictions = model.predict(x_test)
    acc = accuracy_score(y_test,predictions,normalize=True)
    print "accuracy of model is %f" %acc

    f1 = f1_score(y_test,predictions,average='micro')
    print "f1 of model is %f" %f1

    #confusion matrix
    cnf_matrix = confusion_matrix(y_test, predictions)
    print cnf_matrix
    return (acc,f1)

def get_rf_feature_importance(forest,X):
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")

    for f in range(X.shape[1]):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(X.shape[1]), importances[indices],
           color="r", yerr=std[indices], align="center")
    plt.xticks(range(X.shape[1]), indices)
    plt.xlim([-1, X.shape[1]])
    plt.show()
    
def test_model(x_train,y_train,x_test,y_test,model_type):
    #training
    if model_type == "svm":
        model = svm.SVC()
    elif model_type == "naive_bayes":
        model = naive_bayes.GaussianNB()
    elif model_type =="random_forest":
        model = ensemble.RandomForestClassifier(max_depth=50, random_state=0)
    model.fit(x_train, y_train)

    if model_type == "random_forest":
        get_rf_feature_importance(model,x_train)
   
    return evaluate_model(model,x_test,y_test)

def modify_nexts_of_positives(x_train,vectors_to_modify,replacements):
    for i in xrange(0,len(x_train)):
        curr_vec = x_train[i]
        if (vectors_to_modify == curr_vec).all(1).any():
            #find match, and index of match in to modify, and replace
            for rep_ind in xrange(0,len(vectors_to_modify)):
                to_mod = vectors_to_modify[rep_ind]
                if (curr_vec == to_mod).all():
                    break #can probably improve finidng of index
            x_train[i] = replacements[rep_ind]
    return x_train
    
""" methods to call from main """
def load_base_model():
    all_targets = pickle.load(open("../Preprocessing/all_labels.p", "rb"))
    all_targets = [x for x in all_targets if x != "$EOF$"]
    vectors = pickle.load(open("../Models/CV1000vectors.p","rb")) #not includes eof
    return vectors,all_targets

def test_model_cyber(vectors,all_targets, vectors_to_modify, replacements):
    x_train,y_train,x_test,y_test = get_balanced_train(vectors,all_targets)
  
    x_train = modify_nexts_of_positives(x_train,vectors_to_modify,replacements)
    
    print "NB"
    a = test_model(x_train,y_train,x_test,y_test,"naive_bayes")
    print "RF (50)"
    b = test_model(x_train,y_train,x_test,y_test,"random_forest")
    print "SVM"
    c = test_model(x_train,y_train,x_test,y_test,"svm")
    return a,b,c

#gets linked list of actual sentences, not vectors
def get_sents_as_linked_list(all_sents):
    start = Node(all_sents[0])
    curr = start
    for sent in all_sents[1:]:
        curr.reference = Node(sent)
        curr = curr.reference
    return LinkedList(start)

def get_positive_target_nexts(sents_ll,all_sents,sent_vectors,all_targets):
    curr_node = sents_ll.head
    positive_target_nexts = []
    replacements = []
    index_dif = 0
    while (curr_node != None and curr_node.reference !=None):
        if is_eof(curr_node.value):
            index_dif = index_dif + 1
            curr_node = curr_node.reference
            continue
        if not is_eof(curr_node.reference.value):
            curr_sent_index = all_sents.index(curr_node.value)-index_dif
            curr_target = all_targets[curr_sent_index]
            if curr_target == 1:
                curr_vector = sent_vectors[curr_sent_index]
                next_sent_index = all_sents.index(curr_node.reference.value)-index_dif 
                next_vector = sent_vectors[next_sent_index]
                positive_target_nexts.append(next_vector)
                replacements.append(np.mean([curr_vector, next_vector], axis =0))
        curr_node = curr_node.reference
    return positive_target_nexts, replacements

# https://stackoverflow.com/questions/10919664/averaging-list-of-lists-python
def mean(a):
    return sum(a) / len(a)

def main():
    sent_vectors,all_targets = load_base_model()
    all_sents = pickle.load(open("../Preprocessing/grouped_all_sents_no_lemma_lcase.p", "rb"))
    sents_ll = get_sents_as_linked_list(all_sents)

    positive_target_nexts,replacements = get_positive_target_nexts(sents_ll,all_sents,sent_vectors,all_targets)

    test_model_cyber(sent_vectors,all_targets,positive_target_nexts, replacements)


main()
