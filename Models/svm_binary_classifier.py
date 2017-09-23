from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
import numpy as np

def convert_wrongly_predicted_to_sentence(x_test, predictions, y_test):
    false_pos_vectors = []
    false_neg_vectors = []
    correct_vectors = []
    for vector, pred, act in zip(x_test,predictions,y_test):
        if pred > act:
            false_pos_vectors.append(vector)
        elif pred < act:
            false_neg_vectors.append(vector)
        else:
            correct_vectors.append(vector)
    pca = pickle.load(open("100_pca_one_hot_lower_with_punct.p","rb"))

    print "doing inverse transform on all vectors"
    #inverse tranform back to full
    false_pos_full_vectors = []
    false_neg_full_vectors = []
    for vect in false_pos_vectors:
        false_pos_full_vectors.append(pca.inverse_transform(vect))
    for vect in false_neg_vectors:
        false_neg_full_vectors.append(pca.inverse_transform(vect))

    print "turning vectors into sentences"
    word_ind_dict = pickle.load(open("word_ind_dict_lcase.p","rb"))
    ind_word_dict = {v: k for k, v in word_ind_dict.iteritems()}
    #tranform back into original sentences
    false_pos_sentences = []
    false_neg_sentences = []
    for vect in false_pos_full_vectors:
        sentence = []
        for i in xrange(0,len(vect)):
            if i>0:
                #append multiple times for mult occurences of word
                for j in xrange(0,i):
                    sentence.append(ind_word_dict[i])
        false_pos_sentences.append(sentence)
    for vect in false_neg_full_vectors:
        sentence = []
        for i in xrange(0,len(vect)):
            if i>0:
                #append multiple times for mult occurences of word
                for j in xrange(0,i):
                    sentence.append(ind_word_dict[i])
        false_neg_sentences.append(sentence)

    print "false positive sentences:"
    for i in xrange(0,5):
        print false_pos_sentences[i]
    print "false negative sentences:"
    for i in xrange(0,5):
        print false_neg_sentences[i]

    pickle.dump(open(false_pos_sentences, "false_pos_sents_pca_100_lcase.p","wb"))
    pickle.dump(open(false_neg_sentences, "false_neg_sents_pca_100_lcase.p","wb"))
    
def main():
    lab_vectors = pickle.load(open("lab_vectors_one_hot_pca_100_lcase_with_punct.p", "rb"))
    unlab_vectors = pickle.load(open("unlab_vectors_one_hot_pca_100_lcase_with_punct.p", "rb"))
    print type(lab_vectors)
    print len(lab_vectors)
    print len(lab_vectors[0])
    print len(unlab_vectors)
    print len(unlab_vectors[0])
    
    #data and labels
    x = list(lab_vectors)+list(unlab_vectors)
    y = np.zeros((len(x)))
    y[0:len(lab_vectors)] = 1

    #split into train and test
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.3, random_state = 42)

    #training
    clf = svm.SVC()
    clf.fit(x_train, y_train)

    #evaluating
    predictions = clf.predict(x_test)
    acc = accuracy_score(y_test,predictions,normalize=True)
    print "accuracy of model is %f" %acc

    convert_wrongly_predicted_to_sentence(x_test, predictions, y_test)
    
main()

