import pickle
import os

#returns a dict key: file name value: list of list (of words)  
def get_texts(dir_addr,file_names):
    files_sents = {}
    for fname in file_names:
        with open ("%s/%s"%(dir_addr,fname), "r") as myfile:
            all_lines = [line.rstrip() for line in myfile.readlines()] 
        myfile.close()
        unlab_sents_list = []
        lab_sents_list = []
        sentence = []
        labelled = False
        # append the first word of each line to a list
        # when reach a period append that list to sentences_list
        # treat punctuation like words for now
        for line in all_lines:
            split_line = line.split(" ")
            if len(split_line) != 3:
                continue
            word,_,label = split_line
            labelled = labelled or (label!="O")
            if word == ".":
                if labelled:
                    lab_sents_list.append(sentence)
                else:
                    unlab_sents_list.append(sentence)
                sentence = []
                labelled = False
            else:
                sentence.append(word)
        files_sents[fname] = (unlab_sents_list,lab_sents_list)
    return files_sents

def save_sentences_dict(sents):
    pickle.dump(sents, open("sentences_dict_with_punct.p","wb"))

def save_sentences_lists(sents):
    unlab_all = []
    lab_all = []
    for key,value in sents.iteritems():
        unlab_all.extend(value[0])
        lab_all.extend(value[1])
    pickle.dump(unlab_all, open("unlab_sents_with_punct.p","wb"))
    pickle.dump(lab_all, open("lab_sents_with_punct.p","wb"))
    
def main():
    tok_dir = "/home/angel/Documents/Research_287/MalwareTextDB/data/tokenized"
    file_names = os.listdir(tok_dir)
    all_sentences = get_texts(tok_dir,file_names)
    save_sentences_lists(all_sentences)
    
main()
