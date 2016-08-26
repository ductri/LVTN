# -*- coding: utf-8 -*-
"""
Created on Sun Aug 21 21:13:16 2016

@author: ductr
"""
from sklearn import svm
import numpy as np


class SVMClassify:
    def __init__(self, ngram, feature2vectors, training_data, test_data):
        self.__feature2vecs = feature2vectors
        self.__training_data = training_data
        self.__test_data = test_data
        self.__ngram = ngram
        
    def run(self):
        data = self.__ngram.training(self.__training_data['sen'])
        test = self.__ngram.vectorization(self.__test_data['sen'])
        
        training_array = self.__concate(data, self.__vectorization(self.__training_data['sen']))
        test_array = self.__concate(test, self.__vectorization(self.__test_data['sen']))  
        clf = svm.SVC()
        clf.fit(training_array, self.__training_data['lab'])

        score_training = clf.score(training_array, self.__training_data['lab'])
        score_valuation = clf.score(test_array, self.__test_data['lab'])
        return score_training, score_valuation
        
        
    def __concate(self, x, y):
        if y is None:
            return x
        else:
            return np.concatenate((x,y), axis=1)
            
    def __vectorization(self, text_corpus):
        no_feature = len(self.__feature2vecs)
        if no_feature == 0:
            return None
        else:
            data = self.__feature2vecs[0].vectorization(text_corpus)
            for i in range(1, len(self.__feature2vecs)):
                data = np.concatenate(data, self.__feature2vecs[i].vectorization(text_corpus), axis=1)
            return data
            