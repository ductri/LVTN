# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 10:52:22 2016

@author: ductr
"""
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
import nltk
import re

class DataLoader:
    def __init__(self):
        pass
    
    def load(self, is_one_set=False):
        sentences = pd.read_csv("F:\code\python\lvtn\sentence.csv", dtype={'sen':str})
                
        submission = pd.read_csv("F:\code\python\lvtn\submission.csv")
        submission = submission.loc[:, ['sen_id', 'lab']]
        submission.columns = ['id', 'lab']
        data = pd.merge(sentences, submission, on='id')
        data = data[data['nums']!=0].loc[:,['sen','lab']]
        data_size = data.shape[0]
        
        # Just classify 0 and the others
        data.loc[data['lab']!=0,'lab']=1
        
        # Shuffle dataframe
        shuffle_index = np.arange(data_size)
        np.random.shuffle(shuffle_index)
        data = data.iloc[shuffle_index, :]
        data.index = np.arange(data_size)
        
        # Preprocessing
        data['sen'] = self.__preprocessing(data['sen'])
        
        if not is_one_set:
            training_size = data.shape[0]*2/3
            
            training = data[0:training_size]
            test = data[training_size:]
            test.index = np.arange(test.shape[0])
            print 'training_size: '+str(training.shape[0])
            print 'test_size: '+str(test.shape[0])
            return training, test
        else:
            return data
            
    def __preprocessing(self, corpus):
        # Lower text
        corpus = [sen.lower() for sen in corpus]
        corpus = self.__remove_special_char(corpus)
        corpus = self.__label_number(corpus)
        corpus = self.__filter_stopwords(corpus)
        return corpus
        
    def __label_number(self, corpus):
        p=re.compile('([0-9]*([\.])?)[0-9]+')
        corpus = [p.sub('DIGIT', sen) for sen in corpus]
        return corpus
    
    def __remove_special_char(self, corpus):
        p=re.compile('\?|%|@|#|\^|\$')
        return [p.sub('',sen) for sen in corpus]
        
    def __filter_stopwords(self, text_corpus):
        liststopwords = stopwords.words('english')
        temp = [[w for w in nltk.word_tokenize(sen) if w.lower() not in 
        liststopwords] for sen in text_corpus]
        result = [reduce(lambda x,y: x+' '+y, sen) for sen in temp]
        return result