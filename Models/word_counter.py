import pickle
import nltk

def get_num_uniqw(all_sents):
    all_words = []
    duplicate_list = []
    uniq_count = 0
    for sent in all_sents:
        for word in sent:
            try:
                word = word.encode('utf-8')
            except:
                word = word
            if word not in all_words:
                uniq_count = uniq_count +1
                all_words.append(word)
            else:
                if word not in duplicate_list:
                    duplicate_list.append(word)
    num_single_occ = len(all_words) - len(duplicate_list)   
    return uniq_count, num_single_occ, all_words, duplicate_list

def lemmatize_sents(sents):
    nltk.data.path.append("/home/angel/Documents/nltk_data")
    lemma = nltk.wordnet.WordNetLemmatizer()
    new_sents = []
    for sent in sents:
        new_sent = []
        for word in sent:
            try:
                new_sent.append(lemma.lemmatize(word))
            except:
                new_sent.append(word) # if cant lemmatize
        new_sents.append(new_sent)
    return new_sents
            
def main():
    unlab_sents = pickle.load(open("../Preprocessing/unlab_sents_with_punct.p", "rb"))
    lab_sents = pickle.load(open("../Preprocessing/lab_sents_with_punct.p", "rb"))
    
    #lemmatized_unlab_sents = lemmatize_sents(unlab_sents)
    #lemmatized_lab_sents = lemmatize_sents(lab_sents)

    
    all_sents = lemmatized_unlab_sents + lemmatized_lab_sents
    num_uniqw,num_single_occ,all_words,dup_words = get_num_uniqw(all_sents)
    print "number of unique words is: %d" %num_uniqw
    print "number of words that occur only once is: %d" %num_single_occ
    print (all_words)
    
    
main()

