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
    counts_arr = counts.toarray().tolist()
    split_num = int(floor(len(counts_arr)*0.25))
    x_train = counts_arr[:split_num]+counts_arr[-split_num:]
    y_train  =  all_targets[:split_num]+all_targets[-split_num:]
    #print ("x_train len is %d and y_train len is %d " %(len(x_train),len(y_train)))
    x_test = counts_arr[split_num:-split_num]
    y_test  =  all_targets[split_num:-split_num]
    #print ("x_test len is %d and y_test len is %d " %(len(x_test),len(y_test)))
    return x_train,y_train,x_test,y_test

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
    try:
        model = joblib.load(file_name)
        print ("loaded")
    except:
        #training
        if model_type == "svm":
            model = svm.SVC()
        elif model_type == "naive_bayes":
            model = naive_bayes.GaussianNB
        elif model_type =="random_forest":
            model = ensemble.RandomForestClassifier(max_depth=2, random_state=0)
        model.fit(x_train, y_train)
        print ("trained")
        
        #save
        joblib.dump(model,file_name)
        print ("saved")

    evaluate_model(model,x_test,y_test)

def get_sents_and_targets():
    #get list of all words
    unlab_sents = pickle.load(open("../Preprocessing/grouped_unlab_sents_no_lemma_with_punct.p", "rb"))
    lab_sents = pickle.load(open("../Preprocessing/grouped_lab_sents_no_lemma_with_punct.p", "rb"))
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
    for sentence in all_sents:
        curr_vector = np.zeros(100)
        num_words = 0
        for word in all_sents:
            word_vector = np.array(fasttext_model[str(word)])
            curr_vector = curr_vector + word_vector
            num_words = num_words +1
        curr_vector = np.divide(curr_vector,num_words).tolist()
        sent_vectors.append(curr_vector)
    return sent_vectors

def main():
    all_sents, all_targets = get_sents_and_targets()
    save_sents(all_sents,"data.txt")
    fasttext_model = fasttext.skipgram('data.txt','model')
    try:
        sent_vectors = pickle.load(open("sentence_vectors_skipgram_fasttext.p","rb"))
        print ("loaded sentence vectors")
    except:
        print ("creating sentence vectors")
        sent_vectors = get_vector_for_sents(all_sents,fasttext_model)
        print ("created sentence vectors")
        pickle.dump(sent_vectors,open("sentence_vectors_skipgram_fasttext.p","wb"))
        
    test_model(sent_vectors,all_targets,"random_forest",file_name = 'random_forest_model_fasttext_skipgram.p')

main()
