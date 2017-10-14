import numpy as np
from math import floor
import pickle
from sklearn import svm,naive_bayes,ensemble
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def get_balanced_train(counts,all_targets):
    #to get balanced training, unbalanced test - but this isnt randomly selected
    #counts_arr = counts.toarray().tolist()
    counts_arr = counts.tolist()
    split_num = int(floor(len(counts_arr)*0.25))
    x_train = counts_arr[:split_num]+counts_arr[-split_num:]
    y_train  =  all_targets[:split_num]+all_targets[-split_num:]
    print ("x_train len is %d and y_train len is %d " %(len(x_train),len(y_train)))
    x_test = counts_arr[split_num:-split_num]
    y_test  =  all_targets[split_num:-split_num]
    print ("x_test len is %d and y_test len is %d " %(len(x_test),len(y_test)))
    return np.asarray(x_train),np.asarray(y_train),np.asarray(x_test),np.asarray(y_test)


def evaluate_model(model,x_test,y_test):
    #evaluating
    predictions = model.predict(x_test)
    acc = accuracy_score(y_test,predictions,normalize=True)
    print "accuracy of model is %f" %acc

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
    #print x_train.shape
    #print y_train.shape
    model.fit(x_train, y_train)
    #print ("trained")

    evaluate_model(model,x_test,y_test)

def get_sents_and_targets(unlab_sents,lab_sents):
    all_sents_lists = unlab_sents+lab_sents
    all_targets = [0]*len(unlab_sents) + [1]*len(lab_sents)
    return all_sents_lists,all_targets

def main():
    all_sents = pickle.load(open("../Preprocessing/grouped_all_sents_with_lemma_lcase.p", "rb"))
    all_targets = pickle.load(open("../Preprocessing/all_labels.p", "rb"))
    word_dict = pickle.load(open("../Preprocessing/grouped_word_dict_with_lemma_lcase.p", "rb"))
    
    word_ind_dict = {}
    index = 0
    for word in word_dict.keys():
        word_ind_dict[word] = index
        index = index + 1

    print "each sentence is vector of length: %d" %len(word_ind_dict.keys())
    sent_vectors = []
    for sent in all_sents:
        sent_vector =  np.zeros((len(word_ind_dict.keys())))
        for word in sent:
            word = word.lower()
            sent_vector[word_ind_dict[word]] = sent_vector[word_ind_dict[word]]+1
        sent_vectors.append(sent_vector)

    #remove all EOF markers from sent vectors and from targets
    sent_vectors = [x for x in sent_vectors if x != ["$EOF$"]]
    all_targets = [x for x in all_targets if x != "$EOF$"]
    
    print "length of sent vectors after make one hot is"
    print len(sent_vectors)
    print len(sent_vectors[0])
    compon  = 200
    pca = PCA(n_components = compon)
    pca.fit(sent_vectors)

    pca_vectors = pca.transform(sent_vectors)

    print type(pca_vectors)
    print pca_vectors.shape

    print len(all_targets)
    
    x_train,y_train,x_test,y_test = get_balanced_train(pca_vectors,all_targets)
    print "NB"
    test_model(x_train,y_train,x_test,y_test,"naive_bayes")
    print "RF (50)"
    test_model(x_train,y_train,x_test,y_test,"random_forest")
    print "SVM"
    test_model(x_train,y_train,x_test,y_test,"svm")
    
main()
