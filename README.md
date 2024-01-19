# IITK-Scholar
Back end code for iitk scholar portal , query type : Search Professor.
The code uses TFIDF to list important words in the thesis of the research papers done under the supervision of these profs and then does a sparse matrix decomposition to create an efficient embedding for each professor and each important keyword.
The profs are then ranked according to the similarity of these important keywords in their research thesis.

##########
Run the files in the order of :
  prof_test.py
  factor1.py
  query.py


##########
link for the  actual portal is  : http://schinmay.pythonanywhere.com/


########
link for the data : https://drive.google.com/open?id=124M8y8Ky09oQ49jqXUPROJNncQp3WPso

