#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 23:38:38 2020

@author: jotigautam
"""


import nltk
from nltk.corpus import stopwords
import pandas as pd
import re

sms = pd.read_csv("/Users/jotigautam/spyder-py3/smsspamcollection/SMSSpamCollection", sep = '\t', 
                  names= ["type", "message"])

#initializing wordnet lemmatizer
wordnet = nltk.WordNetLemmatizer()
corpus = []

for i in range(len(sms)):
    review = re.sub('[^a-zA-Z]', ' ', sms['message'][i])
    review = review.lower()   
    review = review.split()
    review = [wordnet.lemmatize(word) for word in review if word not in set(stopwords.words('english'))]
    review = " ".join(review)
    corpus.append(review)
    
from sklearn.feature_extraction.text import TfidfVectorizer
tf = TfidfVectorizer(max_features= 5000)
vmatrix = tf.fit_transform(corpus).toarray()

y = pd.get_dummies(sms['type'])
y = y.iloc[:,1].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(vmatrix, y, test_size = 0.20, random_state = 0 ) 


from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(x_train, y_train)

y_check = spam_detect_model.predict(x_test)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_check, y_test)


from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test, y_check)









