# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 22:00:53 2016

@author: ductr
"""

import pandas as pd
import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn import metrics

training_data, test_data, raw_training, raw_test, raw = preprocessing.load()

src = 'F:\\code\\python\\lvtn\\metamap\\metamapQ\\metamapQ\\output.json'
metamap = pd.read_json(src)
x = metamap.data
for i in x:    
    id_sen = raw[raw['sen']==i['sentence']]['id']
    if len(id_sen)==1:
        id_sen.index = [0]
        i['id'] = id_sen[0]

x = [pd.Series(i) for i in x]
x = pd.DataFrame(x)

def trans(x):
    phrase = x['phrase'].replace(' ', '_')
#    if x['negation']:
#        return phrase + ' n_'+phrase
#    else: return phrase
    return phrase

x['ngrams'] = [reduce(lambda x, y: x+' '+y, map(trans, sen)) for sen in x['ngrams']]

training_data = pd.merge(training_data, x, how='inner', on='id')
test_data = pd.merge(test_data, x, how='inner', on='id')

vectorizer = CountVectorizer(min_df=1, decode_error="ignore", analyzer="word", 
                                        lowercase=True, binary=True, ngram_range=(1,2),
                                        stop_words='english')
data_array = vectorizer.fit_transform(training_data['ngrams']).toarray()
data_y = training_data['lab']*1.0

test_x = vectorizer.transform(test_data['ngrams']).toarray()
test_y = test_data['lab']*1.0
score = []
for c in range(10, 500, 1):
    clf = svm.SVC(decision_function_shape='ovr', C=c, kernel='rbf', 
                      class_weight='balanced')
    clf.fit(data_array, data_y)
    predict = clf.predict(test_x)
    s = metrics.precision_score(test_y, predict, average="weighted")
    r = metrics.recall_score(test_y, predict, average="weighted")
    f1 = metrics.f1_score(test_y, predict, average="weighted")
    score.append(f1)
    
    
