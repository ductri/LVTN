# -*- coding: utf-8 -*-
"""
Created on Thu Dec 01 13:14:05 2016

@author: ductr
"""

result = pd.read_csv('F:\code\python\lvtn\Negation-20161130T020606Z\Negation\\result.csv')
result = result.dropna()
negation_id = set()
for i in result.index:
    negation_id.add(result['0'][i])
    
dic = {}
for i in result.index:
    sen = result['7'][i]
    start = sen.find('[PREN]') + 6
    end = sen[start:].find('[PREN]') + start
    dic[result['0'][i]] = sen[start:end]

neg_word = [w.lower() for w in dic.values()]
neg_word = [w.replace('.','') for w in neg_word]
neg_word = [w.replace('-','') for w in neg_word]
negation = pd.DataFrame({'id':dic.keys(), 'neg_word':neg_word})
negation = pd.merge(negation, raw, on='id', how='right')
negation['neg_bin']=0
negation.loc[~negation.neg_word.isnull(),'neg_bin'] = 1