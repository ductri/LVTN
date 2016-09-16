# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 09:42:12 2016

@author: ductr
"""
from sklearn.cross_validation import KFold
import data_loader
def cv(k=5):
    print '*'*70
    print 'Running...'
    score_training = 0
    score_valuation = 0
    sc=0
    re=0
    f1=0
    
    data = data_loader.load(True)
    kf = KFold(data.shape[0], n_folds=k)
    for train_index, test_index in kf:
        data_training = data.iloc[train_index, :]
        test_training = data.iloc[test_index, :]
        print 'train_size: '+str(data_training.shape[0])
        print 'test_size: '+str(test_training.shape[0])
        ngram = N_gram()
        mySVM = SVMClassify(ngram, [], data_training, test_training)
        score_t, score_v, s, r, f = mySVM.run()
        score_training += score_t
        score_valuation += score_v
        sc += s
        re += r
        f1 += f
    score_training /= len(kf)
    score_valuation /= len(kf)
    sc /= 1.0*len(kf)
    re /= 1.0*len(kf)
    f1 /= 1.0*len(kf)
    
    print '*'*70
    print 'Finish!'
    return score_training, score_valuation, sc, re, f1