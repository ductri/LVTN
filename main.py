# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 09:57:36 2016

@author: ductr
"""

from svm import SVMClassify
from ngram import N_gram
from data_loader import DataLoader

data_loader = DataLoader()
training_data, test_data = data_loader.load()

ngram = N_gram()
#ngram.training(training_data['sen'])

mySVM = SVMClassify(ngram, [], training_data, test_data)
x,y = mySVM.run()
#mySVM.evaluate()

def evaluate(no_time=100):
    print '*'*70
    print 'Running...'
    score_training = 0
    score_valuation = 0
    
    for i in range(no_time):
        training_data, test_data = data_loader.load()
        mySVM = SVMClassify(ngram, [], training_data, test_data)
        score_t, score_v = mySVM.run()
        score_training += score_t
        score_valuation += score_v
    score_training /= no_time
    score_valuation /= no_time
    
    print score_training
    print score_valuation
    print '*'*70
    print 'Finish!'
    return score_training, score_valuation