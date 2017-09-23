import pickle
import nltk
import re

def get_num_uniqw(all_sents,lcase=False):
    all_words = []
    duplicate_list = []
    uniq_count = 0
    for sent in all_sents:
        for word in sent:
            try:
                word = word.encode('utf-8')
            except:
                word = word
            if lcase:
                word = word.lower()
            if word not in all_words:
                uniq_count = uniq_count +1
                all_words.append(word)
            else:
                if word not in duplicate_list:
                    duplicate_list.append(word)
    num_single_occ = len(all_words) - len(duplicate_list)   
    return uniq_count, num_single_occ, all_words, duplicate_list

# https://stackoverflow.com/questions/15586721/wordnet-lemmatization-and-pos-tagging-in-python
def get_simple_pos(pos_list):
    nltk_pos_all = []
    num_adj = 0
    num_verb = 0
    num_noun = 0
    num_adv = 0
    num_other = 0
    for sent in pos_list:
        nltk_pos_sent = []
        for pos in sent:
            if pos.startswith('J'):
                nltk_pos_sent.append(nltk.corpus.wordnet.ADJ)
                num_adj = num_adj + 1
            elif pos.startswith("V"):
                nltk_pos_sent.append(nltk.corpus.wordnet.VERB)
                num_verb = num_verb +1
            elif pos.startswith("N"):
                nltk_pos_sent.append(nltk.corpus.wordnet.NOUN)
                num_noun = num_noun +1
            elif pos.startswith("V"):
                nltk_pos_sent.append(nltk.corpus.wordnet.ADV)
                num_verb = num_verb+1
            else:
                nltk_pos_sent.append(nltk.corpus.wordnet.NOUN) #default
                num_other = num_other+1
        nltk_pos_all.append(nltk_pos_sent)
    return nltk_pos_all

def lemmatize_sents(sents,pos_simple):
    lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()
    new_sents = []
    fail_count = 0
    for sent,pos_sent in zip(sents,pos_simple):
        new_sent = []
        for word,word_pos in zip(sent,pos_sent):
            try:
                new_sent.append(lemmatizer.lemmatize(word,pos=word_pos))
            except:
                fail_count = fail_count + 1
                new_sent.append(word) # if cant lemmatize
        new_sents.append(new_sent)
    return new_sents

def group_num(sents,pos_list):
    new_sents = []
    for sent,pos_sent in zip(sents,pos_list):
        new_sent = []
        for word,pos in zip(sent,pos_sent):
            if pos == "CD" or re.search(r'[a-zA-Z]+',word) == None:
                new_sent.append("$NUM$")
            else:
                new_sent.append(word)
        new_sents.append(new_sent)
    return new_sents

def group_md5(sents):
    new_sents = []
    for sent in sents:
        new_sent = []
        for word in sent:
            if len(word) == 32 and re.search(r'^[a-fA-F0-9]+$',word) != None:
                new_sent.append("$MD5$")
            else:
                new_sent.append(word)
        new_sents.append(new_sent)
    return new_sents

def group_doubleslash(sents):
    new_sents = []
    for sent in sents:
        new_sent = []
        for word in sent:
            if re.search(r'\\',word) != None:
                new_sent.append("$doubleSlash$")
            else:
                new_sent.append(word)
        new_sents.append(new_sent)
    return new_sents

def group_genhex(sents):
    new_sents = []
    for sent in sents:
        new_sent = []
        for word in sent:
            if re.search(r'[g-wy-zG-WY-Z]+',word) == None:
                new_sent.append("$HEX$")
            else:
                new_sent.append(word)
        new_sents.append(new_sent)
    return new_sents

def save_processed(sents,fname):
    pickle.dump(sents, open(fname,"wb"))

def main():
    nltk.data.path.append("/home/angel/Documents/nltk_data")
    unlab_sents = pickle.load(open("unlab_sents_with_punct.p", "rb"))
    lab_sents = pickle.load(open("lab_sents_with_punct.p", "rb"))
    unlab_pos = pickle.load(open("unlab_pos_with_punct.p", "rb"))
    lab_pos = pickle.load(open("lab_pos_with_punct.p", "rb"))

    #processed_sents = lemmatize_sents(unlab_sents,get_simple_pos(unlab_pos))
    processed_sents = group_num(unlab_sents,unlab_pos)
    processed_sents = group_md5(processed_sents)
    processed_sents = group_doubleslash(processed_sents)
    processed_sents = group_genhex(processed_sents)
    processed_unlab_sents = processed_sents
    
    #processed_sents = lemmatize_sents(lab_sents,get_simple_pos(lab_pos))
    processed_sents = group_num(lab_sents,lab_pos)
    processed_sents = group_md5(processed_sents)
    processed_sents = group_doubleslash(processed_sents)
    processed_sents = group_genhex(processed_sents)
    processed_lab_sents = processed_sents

    processed_sents = processed_lab_sents+processed_unlab_sents
    num_uniqw,num_single_occ,all_words,dup_words = get_num_uniqw(processed_sents,lcase=False)
    #print all_words
    print "number of unique words is: %d" %num_uniqw
    print "number of words that occur only once is: %d" %num_single_occ

    save_processed(processed_unlab_sents,"grouped_unlab_sents_no_lemma_with_punct.p")
    save_processed(processed_lab_sents,"grouped_lab_sents_no_lemma_with_punct.p")
    
main()

