import pickle
import os
import math
from scipy import spatial
import random
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats.stats import pearsonr   

#returns a list of lists: the major lists have the sentences of the files, and minor lists are the sentences
def get_file_sents(dir_addr,file_names):
    files_sents_list = []
    for fname in file_names:
        with open ("%s/%s"%(dir_addr,fname), "r") as myfile:
            all_lines = [line.rstrip() for line in myfile.readlines()] 
        myfile.close()
        sentence = []
        curr_file = []
        # append the first word of each line to a list
        # when reach a period append that list to sentences_list
        # treat punctuation like words for now
        for line in all_lines:
            split_line = line.split(" ")
            if len(split_line) != 3:
                continue
            word,pos,label = split_line
            if word == ".":
                curr_file.append(sentence)
                sentence = []
            else:
                sentence.append(word)
        files_sents_list.append(curr_file)
    return files_sents_list

def get_all_sents():
    #load sentence vectors that want to use ...
    unlab_sents = pickle.load(open("../Preprocessing/unlab_sents_with_punct.p", "rb"))
    lab_sents = pickle.load(open("../Preprocessing/lab_sents_with_punct.p", "rb"))
    all_sents = unlab_sents + lab_sents
    return all_sents

def get_sent_and_context_lists():
    try:
        curr_sent_list = pickle.load(open("curr_sent_list_with_punct.p","rb"))
        prev_sent_list = pickle.load(open("prev_sent_list_with_punct.p","rb"))
        next_sent_list = pickle.load(open("next_sent_list_with_punct.p","rb"))
    except:
        print ("couldn't load sents and contexts, so making...")
        tok_dir = "/home/angel/Documents/Research_287/MalwareTextDB/data/tokenized"
        file_names = os.listdir(tok_dir)
        files_sents_list = get_file_sents(tok_dir,file_names)
        curr_sent_list = []
        prev_sent_list = []
        next_sent_list = []
        for f in files_sents_list:
            for i in xrange(1,len(f) -1):
                prev_sent_list.append(f[i-1])
                next_sent_list.append(f[i+1])
                curr_sent_list.append(f[i])
            prev_sent_list.append([])
            next_sent_list.append(f[0])
            curr_sent_list.append(f[1])
            prev_sent_list.append(f[len(f)-2])
            next_sent_list.append(f[len(f)-1])
            curr_sent_list.append([])
                                  
        pickle.dump(curr_sent_list,open("curr_sent_list_with_punct.p","wb"))
        pickle.dump(prev_sent_list,open("prev_sent_list_with_punct.p","wb"))
        pickle.dump(next_sent_list,open("next_sent_list_with_punct.p","wb"))    
    return curr_sent_list,prev_sent_list,next_sent_list

def get_dists(curr_sent_list, prev_sent_list,next_sent_list,all_sents,sent_vectors,frac=10000):
    curr_dist_list = []
    prev_dist_list = []
    next_dist_list = []
    i = 0
    for curr_sent1,prev_sent1,next_sent1 in zip(curr_sent_list,prev_sent_list,next_sent_list):
        print ("iteration: %d" %i)
        prev_sent_vect1 = []
        next_sent_vect1 = []
        if len(curr_sent1) == 0: #when curr is not a sentence, skip
            continue
        curr_sent_vect1 = sent_vectors[all_sents.index(curr_sent1)]
        if (len(prev_sent1) > 0):
            prev_sent_vect1 = sent_vectors[all_sents.index(prev_sent1)]
        if (len(next_sent1) >0):
            next_sent_vect1 = sent_vectors[all_sents.index(next_sent1)]

        #COMPARISON SENTENCES
        for curr_sent2,prev_sent2,next_sent2 in zip(curr_sent_list[i:],prev_sent_list[i:],next_sent_list[i:]):
            if random.randint(0,frac) != 17: #only use 1/variable
                continue
            if len(curr_sent2) == 0:
                continue
            curr_sent_vect2 = sent_vectors[all_sents.index(curr_sent2)]
            prev_sent_vect2 = []
            next_sent_vect2 = []
            if (len(prev_sent2) > 0):
                prev_sent_vect2 = sent_vectors[all_sents.index(prev_sent2)]
            if (len(next_sent2) >0):
                next_sent_vect2 = sent_vectors[all_sents.index(next_sent2)]

            #if some prevs or nexts are unavailable skip all of it
            #could still do it for next or prev in future, but for now ....
            if (len(prev_sent_vect1) == 0 or len(prev_sent_vect2) ==  0 or
                len(next_sent_vect1) == 0 or len(next_sent_vect2) == 0):
                continue
            #append the distance between curr to curr_dist_list, and prev to prev_dist_list etc
            curr_dist = 1 - spatial.distance.cosine(curr_sent_vect1,curr_sent_vect2)
            curr_dist_list.append(curr_dist)
            prev_dist = 1 - spatial.distance.cosine(prev_sent_vect1,prev_sent_vect2)
            prev_dist_list.append(prev_dist)
            next_dist =  1 - spatial.distance.cosine(next_sent_vect1,next_sent_vect2)
            next_dist_list.append(next_dist)
  
        i = i+1

    print ("saving dist lists")
    pickle.dump(curr_dist_list, open("some_curr_dists_no_preproc_with_punct.p","wb"))
    pickle.dump(prev_dist_list, open("some_prev_dists_no_preproc_with_punct.p","wb"))
    pickle.dump(next_dist_list, open("some_next_dists_no_preproc_with_punct.p","wb"))
    return curr_dist_list, prev_dist_list,next_dist_list

def graph_curr_context(curr_list,prev_list,next_list):
    plt.scatter(curr_list,prev_list,c="b",alpha=0.5)
    plt.scatter(curr_list,next_list,c="r",alpha=0.5)
    plt.title("Current similarity vs Context similarity")
    plt.xlabel("Sentence Similarity")
    plt.ylabel("Context Similarity")
    plt.show()
    
def main():
    
    try:
        curr_dist_list = pickle.load(open("some_curr_dists_no_preproc_with_punct.p","rb"))
        prev_dist_list = pickle.load(open("some_prev_dists_no_preproc_with_punct.p","rb"))
        next_dist_list = pickle.load(open("some_next_dists_no_preproc_with_punct.p","rb"))
    except:
        print ("failed to load dists lists...")
        curr_sent_list, prev_sent_list,next_sent_list = get_sent_and_context_lists()

        #these are both in corresponding order
        all_sents = get_all_sents()
        sent_vectors = pickle.load(open("no_preproc_sentence_vectors_skipgram_fasttext.p","rb"))

        curr_dist_list, prev_dist_list, next_dist_list = get_dists(curr_sent_list, prev_sent_list,next_sent_list,all_sents,sent_vectors,frac=10000)

    print curr_dist_list
    print prev_dist_list
    print next_dist_list

    #print the correlation between curr and prev, and correlation between curr and next
    print ("correlation with prev similarity:")
    print pearsonr(curr_dist_list,prev_dist_list)
    print ("correlation with next similarity:")
    print pearsonr(next_dist_list,prev_dist_list)
    
    #graph the curr dists against prev dists and next dists
    graph_curr_context(curr_dist_list,prev_dist_list,next_dist_list)


main()
