import numpy as np
import pickle

def main():
    unlab_sents = pickle.load(open("../Preprocessing/grouped_unlab_sents_with_punct.p", "rb"))
    lab_sents = pickle.load(open("../Preprocessing/grouped_lab_sents_with_punct.p", "rb"))
    word_dict = pickle.load(open("../Preprocessing/grouped_lower_word_dict_with_punct.p", "rb"))
    
    word_ind_dict = {}
    index = 0
    for word in word_dict.keys():
        word_ind_dict[word] = index
        index = index + 1
    pickle.dump(word_ind_dict,open("word_ind_dict_lcase.p","wb"))

    print "each sentence is vector of length: %d" %len(word_ind_dict.keys())
    unlab_vectors = []
    lab_vectors = []
    for sent in unlab_sents:
        sent_vector =  np.zeros((len(word_ind_dict.keys())))
        for word in sent:
            word = word.lower()
            sent_vector[word_ind_dict[word]] = sent_vector[word_ind_dict[word]]+1
        unlab_vectors.append(sent_vector)

    for sent in lab_sents:
        sent_vector =  np.zeros((len(word_ind_dict.keys())))
        for word in sent:
            word = word.lower()
            sent_vector[word_ind_dict[word]] = sent_vector[word_ind_dict[word]]+1
        lab_vectors.append(sent_vector)    

    pickle.dump(lab_vectors, open("lab_vectors_one_hot_lower_with_punct.p", "wb"))
    pickle.dump(unlab_vectors, open("unlab_vectors_one_hot_lower_with_punct.p", "wb"))
    
main()
