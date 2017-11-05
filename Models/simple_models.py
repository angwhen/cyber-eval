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
    #print x_train.shape
    #print y_train.shape
    model.fit(x_train, y_train)
    #print ("trained")

    evaluate_model(model,x_test,y_test)
    
'''
def get_sents_and_targets(unlab_sents,lab_sents):
    all_sents_lists = unlab_sents+lab_sents
    all_targets = [0]*len(unlab_sents) + [1]*len(lab_sents)
    return all_sents_lists,all_targets
'''

def bow200():
    all_sents = pickle.load(open("../Preprocessing/grouped_all_sents_with_lemma_lcase.p", "rb"))
    all_targets = pickle.load(open("../Preprocessing/all_labels.p", "rb"))
    word_dict = pickle.load(open("../Preprocessing/grouped_word_dict_with_lemma_lcase.p", "rb"))

    #remove all EOF markers from sent vectors and from targets
    all_sents = [x for x in all_sents if not (len(x) == 1 and x[0] == "$eof$")]
    all_targets = [x for x in all_targets if x != "$EOF$"]

    
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
    
    #x_train,y_train,x_test,y_test = get_balanced_train(pca_vectors,all_targets)
    x_train,y_train,x_test,y_test = get_unbalanced_train(pca_vectors,all_targets)
    print "NB"
    test_model(x_train,y_train,x_test,y_test,"naive_bayes")
    print "RF (50)"
    test_model(x_train,y_train,x_test,y_test,"random_forest")
    print "SVM"
    test_model(x_train,y_train,x_test,y_test,"svm")

    pickle.dump(pca_vectors, open("bow200vectors.p","wb")) #not includes eof

def count_vectorizer():
    all_sents = pickle.load(open("../Preprocessing/grouped_all_sents_no_lemma_lcase.p", "rb"))
    all_targets = pickle.load(open("../Preprocessing/all_labels.p", "rb"))

    #remove all EOF markers from sent vectors and from targets
    all_sents = [x for x in all_sents if not (len(x) == 1 and x[0] == "$eof$")]
    all_targets = [x for x in all_targets if x != "$EOF$"]

    all_sent_strs = []
    for sentence in all_sents:
        sent_str = ""
        for word in sentence:
            sent_str = sent_str + str(word) + " "
        all_sent_strs.append(sent_str)
        
    count_vectorizer = CountVectorizer()
    counts = count_vectorizer.fit_transform(all_sent_strs)
    
    print type(counts)
    print len(all_targets)
    
    counts = counts.toarray().tolist()

    #pickle.dump(counts, open("CVvectors.p","wb"))
    
    pca = PCA(n_components = 1000)
    pca.fit(counts)

    pca_vectors = pca.transform(counts)
    
    pickle.dump(pca_vectors, open("CV1000vectors.p","wb")) #not includes eof
    
    
    x_train,y_train,x_test,y_test = get_balanced_train(pca_vectors,all_targets)
    print "NB"
    test_model(x_train,y_train,x_test,y_test,"naive_bayes")
    print "RF (50)"
    test_model(x_train,y_train,x_test,y_test,"random_forest")
    print "SVM"
    test_model(x_train,y_train,x_test,y_test,"svm")

def get_vector_for_sents(all_sents,fasttext_model):
    sent_vectors = []
    i = 0
    for sentence in all_sents:
        #print "iteration : %d" %i
        #i = i+1
        #print sentence 
        curr_words_list = []
        if (len(sentence)==0 ): #why are there even len 0 sentences
            curr_vector = [0]*100
        else:
            for word in sentence:
                word_vector = fasttext_model[str(word)]
                curr_words_list.append(word_vector)
            curr_vector = np.mean(np.array(curr_words_list),axis=0).tolist()
        #print len(curr_vector)
        sent_vectors.append(curr_vector)
    return sent_vectors

def skipgram_mod():
    import fasttext
    all_sents = pickle.load(open("../Preprocessing/grouped_all_sents_no_lemma_lcase.p", "rb"))
    all_targets = pickle.load(open("../Preprocessing/all_labels.p", "rb"))
    
    fasttext_model = fasttext.skipgram('data.txt','model')

    sent_vectors = get_vector_for_sents(all_sents,fasttext_model)
    pickle.dump(sent_vectors,open("skipgram_mod_vectors.p","wb"))
        
    x_train,y_train,x_test,y_test = get_balanced_train(sent_vectors,all_targets)
    print "NB"
    test_model(x_train,y_train,x_test,y_test,"naive_bayes")
    print "RF (50)"
    test_model(x_train,y_train,x_test,y_test,"random_forest")
    print "SVM"
    test_model(x_train,y_train,x_test,y_test,"svm")
    
def main():
    #bow200()
    count_vectorizer()
    #skipgram_mod()
    
main()
