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
    
""" methods to call from main """
def load_base_model():
    all_targets = pickle.load(open("../Preprocessing/all_labels.p", "rb"))
    all_targets = [x for x in all_targets if x != "$EOF$"]
    vectors = pickle.load(open("../Models/CV1000vectors.p","rb")) #not includes eof
    return vectors,all_targets

def get_sents_as_linked_list(all_sents):
    start = Node(all_sents[0])
    curr = start
    for sent in all_sents[1:]:
        curr.reference = Node(sent)
        curr = curr.reference
    return LinkedList(start)

def get_similarity_list_pairs(sents_ll,all_sents,sent_vectors,all_targets):
    curr_node = sents_ll.head
    pos_similarities = []
    neg_similarities = []
    index_dif = 0
    while (curr_node != None and curr_node.reference !=None):
        if is_eof(curr_node.value):
            index_dif = index_dif + 1
            curr_node = curr_node.reference
            continue
        if not is_eof(curr_node.reference.value):
            curr_sent_index = all_sents.index(curr_node.value)-index_dif
            curr_target = all_targets[curr_sent_index]
            
            curr_vector = sent_vectors[curr_sent_index]
            next_sent_index = all_sents.index(curr_node.reference.value)-index_dif 
            next_vector = sent_vectors[next_sent_index]
            if curr_target == 1:
                pos_similarities.append(1 - spatial.distance.cosine(curr_vector,next_vector))
            else:
                neg_similarities.append(1 - spatial.distance.cosine(curr_vector,next_vector))
        
        curr_node = curr_node.reference
    return pos_similarities,neg_similarities

import scipy as sp
import scipy.stats

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
    return m, m-h, m+h

def graph_histogram(data,graph_title = ""):
    m, lower, upper = mean_confidence_interval(data)
    print "mean is %f, confidence interval (0.95) is (%f,%f)"%(m,lower,upper)
    plt.hist(data, bins=10)

    plt.title(graph_title)
    plt.xlabel('Similarity between adjacent sentences')
    plt.ylabel('Count')
    plt.grid(True)

    plt.show()


def get_targets_in_order(all_sents,sent_vectors,all_targets):
    targets_in_order = []
    for sent in all_sents:
        sent_vectors_index = sent_vectors.index(sent)
    
def main():
    sent_vectors,all_targets = load_base_model()
    #all_sents = pickle.load(open("../Preprocessing/grouped_all_sents_no_lemma_lcase.p", "rb"))
    #sents_ll = get_sents_as_linked_list(all_sents)

    #experiment 5: part 1
    """
    positive_sims, negative_sims =get_similarity_list_pairs(sents_ll,all_sents,sent_vectors,all_targets)
    print "positive similarity average: %.5f" %np.mean(positive_sims)
    print "negative similarity average: %.5f" %np.mean(negative_sims)
    
    graph_histogram(positive_sims,"Positive")
    graph_histogram(negative_sims,"Negative")
    """
    
    #experiment 5: part 2
    #everything already in order
    im = []
    im.append(all_targets)
    im.append(all_targets)
    plt.imshow(im, aspect = "auto", cmap="Blues", interpolation = "nearest")
    plt.show()
main()
