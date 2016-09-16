# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 09:46:33 2016

@author: ductr
"""
from sklearn import svm
import numpy as np
from sklearn import metrics

def run(training_array, label):
    data = ngram.training(training_data['sen'])
    test = ngram.vectorization(test_data['sen'])
    
    training_array = concate(data, vectorization(training_data['sen']))
    test_array = concate(test, vectorization(test_data['sen']))  
    training_data = training_array
    test_data = test_array
    
    clf.fit(training_array, training_data['lab'])

    score_training = clf.score(training_array, training_data['lab'])
    score_valuation = clf.score(test_array, test_data['lab'])
    
    test_predict = clf.predict(test_array)
    
    return score_training, score_valuation, \
        metrics.precision_score(test_data['lab'], test_predict), \
        metrics.recall_score(test_data['lab'], test_predict),\
        metrics.f1_score(test_data['lab'], test_predict)
        
        
def concate(self, x, y):
    if y is None:
        return x
    else:
        return np.concatenate((x,y), axis=1)
        
def vectorization(self, text_corpus):
    no_feature = len(feature2vecs)
    if no_feature == 0:
        return None
    else:
        data = feature2vecs[0].vectorization(text_corpus)
        for i in range(1, len(feature2vecs)):
            data = np.concatenate(data, feature2vecs[i].vectorization(text_corpus), axis=1)
        return data