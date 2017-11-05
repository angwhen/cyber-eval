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

class Node(object):
    def __init__(self,value,next=None):
        self.value = value
        self.reference = next

class LinkedList(object):
    def __init__(self,head_node):
        self.head = head_node
    def get_head(self):
        return self.head
    def get_next(self, curr_node):
        return curr_node.next

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
    print type(counts_arr)
    print type(all_targets)

    split_num = int(floor(len(counts_arr)*0.25))
    x_train = counts_arr[:split_num]+counts_arr[-split_num:]
    y_train  =  all_targets[:split_num]+all_targets[-split_num:]
    print ("x_train len is %d and y_train len is %d " %(len(x_train),len(y_train)))
    x_test = counts_arr[split_num:-split_num]
    y_test  =  all_targets[split_num:-split_num]
    print ("x_test len is %d and y_test len is %d " %(len(x_test),len(y_test)))
    print ("a single vector is %d "%len(x_train[0]))
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

def test_model(x_train,y_train,x_test,y_test,model_type):
    #training
    if model_type == "svm":
        model = svm.SVC()
    elif model_type == "naive_bayes":
        model = naive_bayes.GaussianNB()
    elif model_type =="random_forest":
        model = ensemble.RandomForestClassifier(max_depth=50, random_state=0)
    model.fit(x_train, y_train)
   
    evaluate_model(model,x_test,y_test)

""" methods to call from main """
def load_base_model():
    all_targets = pickle.load(open("../Preprocessing/all_labels.p", "rb"))
    all_targets = [x for x in all_targets if x != "$EOF$"]
    vectors = pickle.load(open("../Models/CV1000vectors.p","rb")) #not includes eof
    return vectors,all_targets

def test_model_cyber(vectors,all_targets):
    x_train,y_train,x_test,y_test = get_balanced_train(vectors,all_targets)
    print "NB"
    test_model(x_train,y_train,x_test,y_test,"naive_bayes")
    print "RF (50)"
    test_model(x_train,y_train,x_test,y_test,"random_forest")
    print "SVM"
    test_model(x_train,y_train,x_test,y_test,"svm")
    
def get_sents_as_linked_list(all_sents):
    start = Node(all_sents[0])
    curr = start
    for sent in all_sents[1:]:
        curr.reference = Node(sent)
        curr = curr.reference
    return LinkedList(start)
        
#should be same index other than eof, so minus one from index of vectors every time get to one
def test_distrib_orig(sents_ll,all_sents,sent_vectors,frac=10000): 
    curr1 = sents_ll.head
    curr_dists = []
    next_dists = []
    curr_sent_pairs = []
    next_sent_pairs = []
    index_dif = 0
    while (curr1!=None):
        if is_eof(curr1.value):
            index_dif = index_dif + 1
            curr1 = curr1.reference
            continue
        curr1_vector = sent_vectors[all_sents.index(curr1.value)-index_dif]
        next1 = curr1.reference
        next1_vector = sent_vectors[all_sents.index(next1.value)-index_dif]
        if is_eof(next1.value):
            curr1 = curr1.reference
            continue
        curr2 = curr1
        index_dif_inner = index_dif
        while (curr2!=None):
            if is_eof(curr2.value):
                index_dif_inner = index_dif_inner + 1
                #print index_dif_inner
                curr2 = curr2.reference
                continue
            if frac != 1 and random.randint(0,frac) != 17: #only use 1/variable
                curr2 = curr2.reference
                continue
            curr2_vector = sent_vectors[all_sents.index(curr2.value)-index_dif_inner]
            next2 = curr2.reference
            next2_vector = sent_vectors[all_sents.index(next2.value)-index_dif_inner]
            if is_eof(next1.value):
                curr2 = curr2.reference
                continue
            if (len(curr1_vector) == 0 or len(curr2_vector) ==  0 or
                len(next1_vector) == 0 or len(next2_vector) == 0):
                curr2 = curr2.reference
                continue
            #remove 0 vectors too
            if (sum(curr1_vector) == 0 or sum(curr2_vector) == 0
                or sum(next1_vector) == 0 or sum(next2_vector) == 0):
                curr2 = curr2.reference
                continue
            curr_dist = 1 - spatial.distance.cosine(curr1_vector,curr2_vector)
            curr_dists.append(curr_dist)
            next_dist =  1 - spatial.distance.cosine(next1_vector,next2_vector)
            next_dists.append(next_dist)
            curr_sent_pairs.append((curr1_vector,curr2_vector))
            next_sent_pairs.append((next1_vector,next2_vector))
            curr2 = curr2.reference
        curr1 = curr1.reference
    pearson_result = pearsonr(curr_dists,next_dists)
    print pearson_result
    return curr_dists,next_dists, curr_sent_pairs, next_sent_pairs, pearson_result[0]

def graph_curr_context(curr_list,next_list):
    plt.scatter(curr_list,next_list,c="r",alpha=0.5)
    plt.title("Current similarity vs Context (Next) similarity")
    plt.xlabel("Sentence Similarity")
    plt.ylabel("Context (Next) Similarity")
    plt.show()

def find_indices_of_worst_correlation(n, curr_dists,next_dists):
    difs = []
    indices = []
    index = 0
    for curr_dist,next_dist in zip(curr_dists,next_dists):
        difs.append(abs(curr_dist - next_dist))
        indices.append(index)
        index = index + 1
    sorted_difs,sorted_indices = zip(*sorted(zip(difs,indices),reverse=True))
    return sorted_indices[:n]
#sent_vectors: numpy array of numpy arrays, value: numpy array
def index_of(sent_vectors,value):
    index = -1
    for sent in sent_vectors:
        index = index + 1 
        for sent_val,val in zip(sent,value):
            if sent_val != val:
                break
        return index
    return -1
            
def get_better_vectors(n,curr_dists,next_dists,curr_sent_pairs,next_sent_pairs,sent_vectors):
    index_list = find_indices_of_worst_correlation(n,curr_dists,next_dists)
    for i in index_list:
        #if curr is more different than next
        #make curr more similar, and next more different
        #make each value 50% closer
        #make each value 50% further
        #ie take difference on each axis, and more each side 25% of the difference closer/further
        vector_length = len(sent_vectors[0]) #all same length
        if curr_dists[i] > next_dists[i]:
            mult = 1
        else:
            mult = -1
        
        curr_vec1 = curr_sent_pairs[i][0]
        curr_vec2 = curr_sent_pairs[i][1]
        assoc_curr_vector_index1 = index_of(sent_vectors,curr_vec1)
        assoc_curr_vector_index2 = index_of(sent_vectors,curr_vec2)
       
        next_vec1 = next_sent_pairs[i][0]
        next_vec2 = next_sent_pairs[i][1]
        assoc_next_vector_index1 = index_of(sent_vectors,next_vec1)
        assoc_next_vector_index2 = index_of(sent_vectors,next_vec2)
        for j in xrange(0,vector_length):
            curr_move = mult*0.25*(curr_vec1[j] - curr_vec2[j])
            next_move = mult*0.25*(next_vec1[j] - next_vec2[j])
            curr_vec1[j] = curr_vec1[j] - curr_move
            curr_vec2[j] = curr_vec2[j] + curr_move
            next_vec1[j] = next_vec1[j] + next_move
            next_vec2[j] = next_vec2[j] - next_move
        sent_vectors[assoc_curr_vector_index1] = curr_vec1
        sent_vectors[assoc_curr_vector_index2] = curr_vec2
        sent_vectors[assoc_next_vector_index1] = next_vec1
        sent_vectors[assoc_next_vector_index2] = next_vec2
    return sent_vectors

def main():
    #load existing model, and then tweak
    sent_vectors,all_targets = load_base_model()
    all_sents = pickle.load(open("../Preprocessing/grouped_all_sents_no_lemma_lcase.p", "rb"))
    times = 0
    while(times <20):
        sents_ll = get_sents_as_linked_list(all_sents)
        print "time: %d "%times
        curr_dists, next_dists, curr_sent_pairs, next_sent_pairs, eval_score = test_distrib_orig(sents_ll,all_sents,sent_vectors,frac=1000)
        curr_dists, next_dists, curr_sent_pairs, next_sent_pairs, eval_score = test_distrib_orig(sents_ll,all_sents,sent_vectors,frac=1000)
        print "number of vectors sampled %d" %len(curr_dists)
        #test_model_cyber(sent_vectors,all_targets)
        #will do so by taking the curr dists and next dists that are the most different?
        #and changing their associated vectors to make them more similar
        #must change those vectors in the original sent_vectors ...
        #preserve order 
        sent_vectors = get_better_vectors(10000,curr_dists,next_dists,curr_sent_pairs,next_sent_pairs,sent_vectors)
        #then make sents_ll again
        #and do test_distrib agin, and loop
        times = times + 1

main()
