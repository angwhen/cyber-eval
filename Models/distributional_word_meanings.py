import pickle
import numpy as np

def main():
    unlab_sents = pickle.load(open("../Preprocessing/grouped_unlab_sents_with_punct.p", "rb"))
    lab_sents = pickle.load(open("../Preprocessing/grouped_lab_sents_with_punct.p", "rb"))
    word_dict = pickle.load(open("../Preprocessing/grouped_lower_word_dict_with_punct.p", "rb"))
    word_ind_dict = pickle.load(open("word_ind_dict_lcase.p", "rb"))

    #go through each sentence in both lab and unlab
    #and for each word entry in dict is one hot vector of size len(word_dict.keys())
    word_nearest_dict = {}
    for word in word_dict.keys():
        word_nearest_dict = np.zeros(len(word_dict.keys()))
                                    
    for sent in unlab_sents+lab_sents:
        for i in xrange(0,len(sent)):
            word = sent[i]
            print word
            if i-1 >= 0:
                prev_word = sent[i-1]
                ind = word_ind_dict[prev_word]
                print ind
                word_nearest_dict[word][ind] = word_nearest_dict[word][ind]+1
            if i-2 >= 0:
                prev2_word = sent[i-2]
                ind = word_ind_dict[prev2_word]
                word_nearest_dict[word][ind] = word_nearest_dict[word][ind]+1
            if i+1 < len(sent):
                next_word = sent[i+1]
                ind = word_ind_dict[next_word]
                print type(ind)
                word_nearest_dict[word][ind] = word_nearest_dict[word][ind]+1
            if i+2 < len(sent):
                next2_word = sent[i+2]
                ind = word_ind_dict[next2_word]
                word_nearest_dict[word][ind] = word_nearest_dict[word][ind]+1
                
    pickle.dump(word_nearest_dict, open("distrib_word_meaning_dict.p","wb"))
        
main()
