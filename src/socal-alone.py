# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 10:55:41 2016

@author: ductr
"""

import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
import numpy as np
from sklearn import metrics
from nltk.stem.snowball import SnowballStemmer
from sklearn.cross_validation import KFold
import sentiwordnet
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import time
import pandas as pd
from sklearn.naive_bayes import BernoulliNB
from sklearn.decomposition import PCA
import re
import json
base_directory = "F:\\code\\python\\lvtn\\"
socal = pd.read_csv(base_directory + "so-cal.csv")
predict = []
for s in socal['socal']:
    if s<-0:
        predict.append(0)
    elif s<0.5:
        predict.append(1)
    else: predict.append(2)
test_y = socal['lab']
a = metrics.accuracy_score(test_y, predict)
s = metrics.precision_score(test_y, predict, average="weighted")
r = metrics.recall_score(test_y, predict, average="weighted")
f1 = metrics.f1_score(test_y, predict, average="weighted")
f1_all = metrics.f1_score(test_y, predict, average=None)
target_names = ['class 0', 'class 1', 'class 2']
print(classification_report(test_y, predict, target_names=target_names))

#Analyst
pos=socal[socal['lab']==2]
neu=socal[socal['lab']==1]
neg=socal[socal['lab']==0]



plt.figure()
plt.plot(pos.socal, np.zeros(pos.shape[0])+2, 'ro', neu.socal, np.zeros(neu.shape[0])+1, 'bo', neg.socal, np.zeros(neg.shape[0]), 'go')
plt.axis([-5, 5, -0.5, 2.5])

plt.figure('log')
plt.plot(f(pos.socal), np.zeros(pos.shape[0])+2, 'ro', f(neu.socal), np.zeros(neu.shape[0])+1, 'bo', f(neg.socal), np.zeros(neg.shape[0]), 'go')
plt.axis([-5, 5, -0.5, 2.5])

pos_score = map(score_sen, pos.sen)
neu_score = map(score_sen, neu.sen)
neg_score = map(score_sen, neg.sen)
plt.plot(pos_score, np.zeros(pos.shape[0])+2, 'ro', neu_score, np.zeros(neu.shape[0])+1, 'bo', neg_score, np.zeros(neg.shape[0]), 'go')

def remove_some(corpus):
    x = preprocessing.lemmatization(corpus)
    x = preprocessing.stemming(x)
    def remove(word):
        if word in BAD or word in GOOD or word in MORE or word in LESS:
            print 'remove'
            return '--'
        else: return word
    temp = [[remove(w) for w in sen.split(' ')] for sen in x]
    result = [reduce(lambda x,y: x+' '+y, sen) for sen in temp]
    return result
    

x = socal[socal.socal<0]['socal'] - 1
socal.loc[socal.socal<0, 'socal']  = x
