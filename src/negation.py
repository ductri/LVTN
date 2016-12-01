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
effected_words = {}
for i in result.index:
    sen = result['7'][i]
    start = sen.find('[PREN]') + 6
    end = sen[start:].find('[PREN]') + start
    #Gia su moi cau chi co 1 NEG, 651 co 2 negation
    dic[result['0'][i]] = sen[start:end]

for key in dic.keys():
    effected_words[key] = []

neg_word = [w.lower() for w in dic.values()]
neg_word = [w.replace('.','') for w in neg_word]
neg_word = [w.replace('-','') for w in neg_word]
negation = pd.DataFrame({'id':dic.keys(), 'neg_word':neg_word})
negation = pd.merge(negation, raw, on='id', how='right')
negation['neg_bin']=0
negation.loc[~negation.neg_word.isnull(),'neg_bin'] = 1

temp = result[result['6']=='negated']
for i in temp.index:
    effected_words[temp['0'][i]].append(temp['1'][i])
x=pd.DataFrame({'id':effected_words.keys(), 'effected_words':effected_words.values()})

effected_words_std = []
for words in list(effected_words):
    if type(words) == float:
        effected_words_std.append(words)
    else:
        print words
        list_words = []
        for w in words:
            list_words.extend(w.split(' '))
        list_words = pp.stemming(pp.lemmatization(list_words))
        effected_words_std.append(list_words)
        
        