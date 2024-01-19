import numpy as np
import csv
import re
import pickle
import imp
import time
start = time.clock()
 
np.set_printoptions(threshold=np.nan)
 
with open('W_100_prof.pickle', 'rb') as p_f:
    w_100= pickle.load(p_f)   # train_features matrix decomposed in to N x K
 
with open('W_p_100_for_prof.pickle', 'rb') as p_f:
    w_p_100 =pickle.load(p_f)  # test_features matrix decomposed similarly
 
with open('train_prof.pickle', 'rb') as p_f:
    train_prof =pickle.load(p_f)  # name of training profs

with open('test_prof.pickle', 'rb') as p_f:
    test_prof =pickle.load(p_f)  # name of testing profs

#test_prof = np.asmatrix(test_prof)
n=len(test_prof)

pred_100 = np.dot(w_100,np.transpose(w_p_100))
pred_100 = np.transpose(pred_100)

#print(test_prof)


for i in range(0,n):
	print(i)
	print("Query :")
	print(test_prof[i])
	idx_100 = (-pred_100[i]).argsort()[:10]
	print("Results :")
	for j in range(0,10):
		print(j+1,train_prof[idx_100[j]])

print (time.clock() - start)