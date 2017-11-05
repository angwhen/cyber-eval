import pickle
import os
import math
from scipy import spatial
import random
import matplotlib.pyplot as plt
import numpy as np
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

            
def get_sents_as_linked_list(all_sents):
    start = Node(all_sents[0])
    curr = start
    for sent in all_sents[1:]:
        curr.reference = Node(sent)
        curr = curr.reference
    return LinkedList(start)

def is_eof(l):
    if (len(l) == 1 and l[0] == "$eof$"):
        return True
    else:
        False
        
#should be same index other than eof, so minus one from index of vectors every time get to one
def get_dists(sents_ll,all_sents,sent_vectors,frac=10000): 
    curr1 = sents_ll.head
    curr_dists = []
    next_dists = []
    index_dif = 0
    while (curr1!=None):
        if is_eof(curr1.value):
            index_dif = index_dif + 1
            curr1 = curr1.reference
            continue
        #print curr1.value
        #print all_sents.index(curr1.value)
        curr1_vector = sent_vectors[all_sents.index(curr1.value)-index_dif]
        next1 = curr1.reference
        #print next1.value
        #print all_sents.index(next1.value)
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
            if random.randint(0,frac) != 17: #only use 1/variable
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
            curr2 = curr2.reference
        curr1 = curr1.reference
    return curr_dists,next_dists

def graph_curr_context(curr_list,next_list):
    plt.scatter(curr_list,next_list,c="r",alpha=0.5)
    plt.title("Current similarity vs Context (Next) similarity")
    plt.xlabel("Sentence Similarity")
    plt.ylabel("Context (Next) Similarity")
    plt.show()
    
def main():
    all_sents = pickle.load(open("../Preprocessing/grouped_all_sents_no_lemma_lcase.p", "rb"))
    sents_ll = get_sents_as_linked_list(all_sents)
    sent_vectors = pickle.load(open("CV1000vectors.p","rb"))
    print len(all_sents)
    print len(sent_vectors)

    curr_dist_list,next_dist_list = get_dists(sents_ll,all_sents,sent_vectors,frac=10000)
    #print curr_dist_list
    #print next_dist_list
    print ("correlation with next similarity:")
    print pearsonr(next_dist_list,curr_dist_list)
    
    #graph the curr dists against prev dists and next dists
    graph_curr_context(curr_dist_list,next_dist_list)


main()
