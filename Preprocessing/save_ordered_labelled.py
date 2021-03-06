import pickle
import os

#returns a dict key: file name value: list of list (of words)  
def get_texts(dir_addr,file_names):
    files_sents = {}
    for fname in file_names:
        with open ("%s/%s"%(dir_addr,fname), "r") as myfile:
            all_lines = [line.rstrip() for line in myfile.readlines()] 
        myfile.close()
        all_sents_list = []
        all_pos_list = []
        all_labels_list = []
        sentence = []
        curr_pos_list = []
        labelled = False
        # append the first word of each line to a list
        # when reach a period append that list to sentences_list
        # treat punctuation like words for now
        for line in all_lines:
            split_line = line.split(" ")
            if len(split_line) != 3:
                continue
            word,pos,label = split_line
            labelled = labelled or (label!="O")
            if word == ".":
                all_sents_list.append(sentence)
                all_pos_list.append(curr_pos_list)
                if labelled:
                    all_labels_list.append(1)
                else:
                    all_labels_list.append(0)
                sentence = []
                curr_pos_list = []
                labelled = False
            else:
                sentence.append(word)
                curr_pos_list.append(pos)
        files_sents[fname] = (all_sents_list,all_pos_list,all_labels_list)
    return files_sents

def save_sentences_lists(sents):
    all_sents = []
    all_pos = []
    all_labels = []
    for key,value in sents.iteritems():
        all_sents.extend(value[0])
        all_sents.append(["$EOF$"]) #to denote end of file
        all_pos.extend(value[1])
        all_pos.append("$EOF$")
        all_labels.extend(value[2])
        all_labels.append("$EOF$")
        
    pickle.dump(all_sents, open("all_sents.p","wb"))
    pickle.dump(all_pos, open("all_pos.p","wb"))
    pickle.dump(all_labels, open("all_labels.p","wb"))
    
    
def main():
    #tok_dir = "/home/angel/Documents/Research_287/MalwareTextDB/data/tokenized"
    tok_dir = "../../MalwareTextDB/data/tokenized"
    file_names = os.listdir(tok_dir)
    all_sentences = get_texts(tok_dir,file_names)
    save_sentences_lists(all_sentences)
    
main()
