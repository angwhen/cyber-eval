import numpy as np
from math import floor
import pickle
from sklearn import svm,naive_bayes,ensemble
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.externals import joblib
import fasttext

def get_balanced_train(counts,all_targets):
    #to get balanced training, unbalanced test - but this isnt randomly selected
    #counts_arr = counts.toarray().tolist()
    counts_arr = counts
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

def test_model(counts,all_targets,model_type,file_name):
    x_train,y_train,x_test,y_test = get_balanced_train(counts,all_targets)
  
    #training
    if model_type == "svm":
        model = svm.SVC()
    elif model_type == "naive_bayes":
        model = naive_bayes.GaussianNB()
    elif model_type =="random_forest":
        model = ensemble.RandomForestClassifier(max_depth=50, random_state=0)
    print x_train.shape
    print y_train.shape
    model.fit(x_train, y_train)
    print ("trained")
        
    #save
    joblib.dump(model,file_name)
    print ("saved")

    evaluate_model(model,x_test,y_test)

def get_sents_and_targets():
    #get list of all words
    unlab_sents = pickle.load(open("../Preprocessing/unlab_sents_with_punct.p", "rb"))
    lab_sents = pickle.load(open("../Preprocessing/lab_sents_with_punct.p", "rb"))
    all_sents_lists = unlab_sents+lab_sents
    all_targets = [0]*len(unlab_sents) + [1]*len(lab_sents)
    return all_sents_lists,all_targets

#possible issue is that sentences are not all in doc order
#so between sentences meaning not preserved
def save_sents(all_sents,fname):
    my_doc = ""
    for sentence in all_sents:
        for word in sentence:
            my_doc = my_doc + str(word) +" "
        my_doc = my_doc + "\n"
    with open(fname, "w") as text_file:
        text_file.write(my_doc)

def get_vector_for_sents(all_sents,fasttext_model):
    sent_vectors = []
    i = 0
    for sentence in all_sents:
        print "iteration : %d" %i
        i = i+1
        print sentence 
        curr_words_list = []
        if (len(sentence)==0 ): #why are there even len 0 sentences
            curr_vector = [0]*100
        else:
            for word in sentence:
                word_vector = fasttext_model[str(word)]
                curr_words_list.append(word_vector)
            curr_vector = np.mean(np.array(curr_words_list),axis=0).tolist()
        print len(curr_vector)
        sent_vectors.append(curr_vector)
    return sent_vectors

def main():
    all_sents, all_targets = get_sents_and_targets()
    try:
        print ("trying to load sentence vectors")
        sent_vectors = pickle.load(open("no_preproc_sentence_vectors_skipgram_fasttext.p","rb"))
        print ("loaded sentence vectors")
    except:
        print ("nevermind")
        save_sents(all_sents,"data.txt")
        print ("making fasttext model")
        fasttext_model = fasttext.skipgram('data.txt','model')
        print ("creating sentence vectors")
        sent_vectors = get_vector_for_sents(all_sents,fasttext_model)
        print ("created sentence vectors")
        pickle.dump(sent_vectors,open("no_preproc_sentence_vectors_skipgram_fasttext.p","wb"))
        
    test_model(sent_vectors,all_targets,"svm",file_name = 'svm_no_preproc_fasttext_skipgram.p')

main()
