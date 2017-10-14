import pickle
import matplotlib.pyplot as plt
import numpy as np

def word_length_hist(words):
    word_lens = [len(w) for w in words]
    print "mean word length: %f, std dev: %f" %(np.mean(np.array(word_lens)),np.std(np.array(word_lens)))
    fig, ax = plt.subplots()
    plt.title("Word Lengths Distribution")
    plt.ylabel("Count")
    plt.xlabel("Word Length")
    plt.hist(word_lens,20)
    plt.show()
    
def most_common_words(word_dict):
    sorted_words = sorted(word_dict, key=word_dict.get, reverse=True)
    for word in sorted_words[:30]:
        print "(%s,%d) " % (word, word_dict[word]),
    
def plot_words_count(word_dict):
    word_counts = word_dict.values() #loses the associated act word
    word_counts.sort(reverse=True) 
    print "number of unique words is: %d" %len(word_counts)

    fig, ax = plt.subplots()
    plt.title("Word Counts")
    #plt.yscale("log")
    plt.ylabel("Count")
    plt.xlabel("Words")
    plt.bar(np.arange(len(word_counts)), word_counts, 1, color='r',edgecolor="none")
    plt.show()
    
def get_words_dict(all_words, lower = False):
    print (type(all_words))
    #make histogram with count of all words
    word_dict = {}
    for word in all_words:
        word = str(word)
        if lower:
            word = word.lower()
        if word in word_dict.keys():
            word_dict[word] = word_dict[word]+1
        else:
            word_dict[word] = 1
    return word_dict
    
def main():
    #load all words, with some words transformed to group
    all_sents = pickle.load(open("grouped_all_sents_no_lemma_lcase.p", "rb"))
  
    all_words = [item for sublist in all_sents for item in sublist]
    print "all words: %d" %(len(all_words))

    lower = True
    word_dict = get_words_dict(all_words, lower)
    pickle.dump(word_dict, open("grouped_word_dict_no_lemma_lcase.p", "wb"))

    
    #word_dict = pickle.load(open("grouped_word_dict_no_lemma_lcase.p", "rb"))

    #plot_words_count(word_dict)
    #most_common_words(word_dict)
    #word_length_hist(word_dict.keys())
    
main()
