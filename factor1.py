import numpy as np
import csv
import re
import pickle
import imp
import time
start = time.clock()
 
np.set_printoptions(threshold=np.nan)
 
with open('theses-corpus.pkl', 'rb') as p_f:
    data = pickle.load(p_f)
 
with open('train_features_for_prof.pickle', 'rb') as p_f:
    train_data = pickle.load(p_f)
 
with open('train_prof.pickle', 'rb') as p_f:
	train_prof = pickle.load(p_f)

with open('test_features_for_prof.pickle', 'rb') as p_f:
	test_data  = pickle.load(p_f)


from sklearn.decomposition import NMF
model = NMF(n_components=100, init='random', random_state=0)
print("Start Transform")
W = model.fit_transform(train_data)
print("Transform done")
H = model.components_
#print(H)
W_p = model.transform(test_data)


 
pickle.dump(W, open("W_100_prof.pickle", "wb"))
pickle.dump(H, open("H_100_prof.pickle", "wb"))
pickle.dump(W_p, open("W_p_100_for_prof.pickle", "wb"))
 
print (time.clock() - start)