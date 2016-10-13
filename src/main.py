# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 09:40:43 2016

@author: ductr
"""

#from svm import SVMClassify
import sys
reload(sys)
sys.setdefaultencoding('utf8')
import preprocessing

from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
import numpy as np
from sklearn import metrics
from nltk.stem.snowball import SnowballStemmer
from sklearn.cross_validation import KFold
import sentiwordnet
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import time
import pandas as pd
from sklearn.naive_bayes import BernoulliNB
from sklearn.decomposition import PCA

base_directory = "F:\\code\\python\\lvtn\\"

# 0:NEG, 1:NEU, 2:POS
training_data, test_data, raw_training, raw_test, raw = preprocessing.load()

def training_ngram(corpus):
    vectorizer = CountVectorizer(min_df=3, decode_error="ignore", analyzer="word", 
                                        lowercase=True, binary=True, ngram_range=(1,2),
                                        stop_words='english')
    data_array = vectorizer.fit_transform(corpus).toarray()
    #print data_array.shape
    return data_array, vectorizer
def training_change_phrase(corpus):
    BAD = ["suffer", "adverse", "hazards", "risk", "death", "insufficient",
           "infection", "recurrence", "restlessness", "mortality", "hazard",
           "chronic", "pain", "negative", "severity"]
    GOOD = ["benefit", "improvement", "advantage", "accuracy", "great",
            "effective", "support", "potential", "superior", "mild", "achieved",
           "Supplementation", "beneficial", "positive"]    
    MORE = ["enhance", "higher", "exceed", "increase", "improve", "somewhat",
            "quite", "very", "higher", "more", "augments", "highest"]
    LESS = ["reduce", "decline", "fall", "less", "little", "slightly", "only", 
            "mildly", "smaller", "lower", "reduction"]
    
    stemmer = SnowballStemmer("english")
    BAD = [stemmer.stem(w) for w in BAD]
    GOOD = [stemmer.stem(w) for w in GOOD]
    MORE = [stemmer.stem(w) for w in MORE]
    LESS = [stemmer.stem(w) for w in LESS]
    
    def sen2vec(sen):
        words = sen.split(' ')
        vecs = [0,0,0,0] #MORE GOOD, MORE BAD, LESS GOOD, LESS BAD
        for i in range(len(words)):
            if words[i] in MORE:
                for k in range(i, len(words)):
                    if words[k] in GOOD:
                        vecs[0] = 1
                        break
                    if words[k] in BAD:
                        vecs[1] = 1
                        break
            elif words[i] in LESS:
                for k in range(i, len(words)):
                    if words[k] in GOOD:
                        vecs[2] = 1
                        break
                    if words[k] in BAD:
                        vecs[3] = 1
                        break
        return vecs
    result = [sen2vec(sen) for sen in corpus]
    return result




def normalize(data):
    ##print 'data:'+str(len(data))
    scale = np.max(np.abs(data), axis=0)
    
    def conv(x):
        if x==0:
            return 1
        else:
            return x
    try:
        scale = map(conv, scale)
    except:
        ##print 'except'
        ##print 'scale=' + str(scale)
        ave = np.average(data, axis=0)
        return (data-ave)/scale
    ave = np.average(data, axis=0)
    return (data-ave)/scale
    
def run(training_data, test_data, raw_training, raw_test, c):
    ##print 'training_data:'+str(training_data.shape)
    data_x, ngram = training_ngram(training_data['sen'])
    ##print 'data_x1:'+str(data_x.shape)
    data_x = normalize(data_x)
    ##print 'data_x2:'+str(data_x.shape)
    ##print 'raw_data:'+str(raw_training.shape)
    data_y = training_data['lab']*1.0
    test_x = ngram.transform(test_data['sen']).toarray()
    test_x = normalize(test_x)
    test_y = test_data['lab']*1.0
    
    #data_x = np.concatenate((data_x, training_change_phrase(training_data['sen'])), axis=1)
    #test_x = np.concatenate((test_x, training_change_phrase(test_data['sen'])), axis=1)
##    
##    #SOCAL
    #socal = pd.read_csv(base_directory + "so-cal.csv")
    #temp = pd.merge(training_data, socal, on='id')
    #data_x = np.concatenate((data_x, np.array(temp['socal']).reshape((temp.shape[0],1))), axis=1)
    
    #temp = pd.merge(test_data, socal, on='id')
    #test_x = np.concatenate((test_x, np.array(temp['socal']).reshape((temp.shape[0],1))), axis=1)
    
    #10
    clf = svm.SVC(decision_function_shape='ovr', C=c, kernel='rbf', class_weight='balanced')
    #clf = svm.NuSVC(nu=c, decision_function_shape='ovr')
    #0.4->0.5
    #clf = BernoulliNB(alpha=c)
    

    
    
    clf.fit(data_x, data_y)
    predict = clf.predict(test_x)
    s = metrics.precision_score(test_y, predict, average="weighted")
    r = metrics.recall_score(test_y, predict, average="weighted")
    f1 = metrics.f1_score(test_y, predict, average="weighted")
    
    f1_all = metrics.f1_score(test_y, predict, average=None)
    
    #print f1
    return predict, s, r, f1, f1_all

def cv(k, c):

    s=0
    r=0
    f1=0
    f1_all = np.zeros(3)
    
    data, data_raw = preprocessing.load(True)
    kf = KFold(data.shape[0], n_folds=k)
    for train_index, test_index in kf:
        training_data = data.iloc[train_index, :]
        raw_training = data_raw.iloc[train_index, :]
        test_data = data.iloc[test_index, :]
        raw_test = data_raw.iloc[test_index, :]
        #print 'train_size: '+str(training_data.shape[0])
        #print 'test_size: '+str(test_data.shape[0])
        
        predict, s_, r_, f1_,f1_all_ = run(training_data, test_data,raw_training, raw_test,  c)

        s += s_
        r += r_
        f1 += f1_
        f1_all += f1_all_
  
    s /= 1.0*len(kf)
    r /= 1.0*len(kf)
    f1 /= 1.0*len(kf)
    f1_all /= len(kf)

    return s, r, f1, f1_all
    
def test(n=50, c=30.5, k=10):
    f1 = 0
    f1_all = 0
    for i in range(0,n):
        _,_,f1_, f1_all_ = cv(k, c)
        f1+= f1_
        f1_all += f1_all_
        
    return f1/n, f1_all/n

def findc(title):
    start = time.time()
    score = []
    score0 = []
    score1 = []
    score2 = []
    c = np.arange(10, 20, 1)
    index = 0
    for i in c:
        print '-'*30
        print(str(index)+'/'+str(len(c)))
        print('c='+str(i))
        print '-'*30
        index = index + 1
        result = test(40, i)
        score.append(result[0])
        score0.append(result[1][0])
        score1.append(result[1][1])
        score2.append(result[1][2])
    plt.figure()
    l1, l2, l3, l4 = plt.plot(c, score, 'b-', c, score0, 'g--', c, score1, 'r--', c, score2, 'b--')
    plt.title(title)
    plt.xlabel('c')
    plt.ylabel('score f1')
    plt.legend([l1, l2, l3, l4], ['Average', 'Negative', 'Neutral', 'Positive'], loc=4, bbox_to_anchor=(1, 1))
    end = time.time()
    print('time: ' + str(end-start))
