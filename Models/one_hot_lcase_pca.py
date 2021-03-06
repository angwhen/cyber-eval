import pickle
import numpy as np
from sklearn.decomposition import PCA

def main():
    lab_vectors = pickle.load(open("lab_vectors_one_hot_lower_with_punct.p", "rb"))
    unlab_vectors = pickle.load(open("unlab_vectors_one_hot_lower_with_punct.p", "rb"))

    compon  = 100
    pca = PCA(n_components = compon)
    pca.fit(lab_vectors+unlab_vectors)

    #can pca so can use to reverse later?
    pickle.dump(pca, open("%d_pca_one_hot_lower_with_punct.p"%compon,"wb"))
    
    pca_lab_vectors = pca.transform(lab_vectors)
    pca_unlab_vectors = pca.transform(unlab_vectors)

    print len(pca_lab_vectors)
    print len(pca_lab_vectors[0])
    
    pickle.dump(pca_lab_vectors, open("lab_vectors_one_hot_pca_%d_lcase_with_punct.p"%compon, "wb"))
    pickle.dump(pca_unlab_vectors, open("unlab_vectors_one_hot_pca_%d_lcase_with_punct.p"%compon, "wb"))
    

main()
