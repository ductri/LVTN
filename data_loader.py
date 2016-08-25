# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 10:52:22 2016

@author: ductr
"""
import pandas as pd
import numpy as np

class DataLoader:
    def __init__(self):
        pass
    
    def load(self):
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
        
        training_size = data.shape[0]*2/3
        
        training = data[0:training_size]
        test = data[training_size:]
        test.index = np.arange(test.shape[0])
        
        #        ng = ngram()
        #        training_vector, vectorizer = ng.vectorization(training['sen'])
        #        
        #        test_vector = vectorizer.transform(test['sen'])
        
        return training, test