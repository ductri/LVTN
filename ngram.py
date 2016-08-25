# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 09:03:16 2016

@author: ductr
"""
from sklearn.feature_extraction.text import CountVectorizer
from feature2vector import Feature2Vector

class N_gram(Feature2Vector):
    
    def __init__(self):
        self.__vectorizer = 0
        self.vectorizer = self.__vectorizer
        pass

    def training(self, corpus):
      #  corpus = []
#        for sen in sentences_db:
#            corpus.append(sen)
        
        self.__vectorizer = CountVectorizer(min_df=2, decode_error="ignore", analyzer="word", 
                                            lowercase=True, binary=True, ngram_range=(1,3),
                                            stop_words='english')
        self.vectorizer = self.__vectorizer
        data_array = self.__vectorizer.fit_transform(corpus).toarray()
        print data_array.shape
        return data_array 
        
    # Override
    def vectorization(self, corpus):
        return self.__vectorizer.transform(corpus).toarray()
    
    
        