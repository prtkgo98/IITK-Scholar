import numpy as np
import csv
import re
import pickle
import imp
 
np.set_printoptions(threshold=np.nan)
 
with open('theses-corpus.pkl', 'rb') as p_f:
    data = pickle.load(p_f)

for i in data:
    if(i[7].lower()=="abstract"):
        i[7]=""

for i in data:
    i[6]=i[6].replace(';','')

for i in data:
	i[5]=i[5].replace(',','')
	i[5]=i[5].replace('.','')

data.sort( key = lambda x : x[5])

for i in data:
    for j in range(0,5):
        i[7]=i[7]+" "+i[6]
    for l in range(0,3):
        i[7]=i[7]+" "+i[2]


n=len(data)
train_sentences=[]
train_prof =[]
test_prof=[]
test_sentences=[]

count =0
i=0
while (i < 15000):
	name = data[i][5]
	index = i+1
	while(data[index][5] == name):
		data[i][7] = data[i][7] + data[index][7]
		index = index + 1	
	train_sentences.append(data[i][7])
	train_prof.append(data[i][5])
	i = index
	count = count + 1


i = 15000
while ( i < 16000):
	#print(i)
	name = data[i][5]
	index = i+1
	while(data[index][5] == name):
		data[i][7] = data[i][7] + data[index][7]
		index = index + 1	
	test_sentences.append(data[i][7])
	test_prof.append(data[i][5])
	i = index
	count = count + 1

ignore_words = ["ourselves", "hers", "between", "yourself", "but", "again", "there", "about", "once", "during", "out", "very", "having", "with", "they", "own", "an", "be", "some", "for", "do", "its", "yours", "such", "into", "of", "most", "itself", "other", "off", "is", "s", "am", "or", "who", "as", "from", "him", "each", "the", "themselves", "until", "below", "are", "we", "these", "your", "his", "through", "don", "nor", "me", "were", "her", "more", "himself", "this", "down", "should", "our", "their", "while", "above", "both", "up", "to", "ours", "had", "she", "all", "no", "when", "at", "any", "before", "them", "same", "and", "been", "have", "in", "will", "on", "does", "yourselves", "then", "that", "because", "what", "over", "why", "so", "can", "did", "not", "now", "under", "he", "you", "herself", "has", "just", "where", "too", "only", "myself", "which", "those", "i", "after", "few", "whom", "t", "being", "if", "theirs", "my", "against", "a", "by", "doing", "it", "how", "further", "was", "here", "than", "a" , "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"]

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(analyzer = "word", stop_words = ignore_words)
print('Start Fit vectorizer')

tfidf = vectorizer.fit(train_sentences)
print('Fit vectorizer') 
 
# print(len(vectorizer.get_feature_names()))
 
print('Start transform train comments')
 
train_features = vectorizer.transform(train_sentences)
 
print('Transformed train comments')
 
print('Start transform test comments')
 
test_features = vectorizer.transform(test_sentences)
 
print('Transformed test comments')
 
# pickle.dump(train_sentences, open("train.pickle", "wb"))
# pickle.dump(test_sentences, open("test.pickle", "wb"))
# pickle.dump(tfidf, open("tfidf.pickle", "wb"))

pickle.dump(train_features, open("train_features_for_prof.pickle", "wb"))
pickle.dump(train_prof, open("train_prof.pickle", "wb"))
pickle.dump(test_features, open("test_features_for_prof.pickle", "wb"))
pickle.dump(test_prof, open("test_prof.pickle", "wb"))