# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 09:40:43 2016

@author: ductr
"""

#from svm import SVMClassify
import preprocessing
import sys  
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
import numpy as np
from sklearn import metrics
from nltk.stem.snowball import SnowballStemmer
from sklearn.cross_validation import KFold
import sentiwordnet
from sklearn.metrics import confusion_matrix

reload(sys)  
sys.setdefaultencoding('utf8')

# 0:NEG, 1:NEU, 2:POS
training_data, test_data, raw_training, raw_test = preprocessing.load()

def training_ngram(corpus):
    vectorizer = CountVectorizer(min_df=2, decode_error="ignore", analyzer="word", 
                                        lowercase=True, binary=True, ngram_range=(1,3),
                                        stop_words='english')
    data_array = vectorizer.fit_transform(corpus).toarray()
    print data_array.shape
    return data_array, vectorizer
def training_change_phrase(corpus):
    BAD = ["suffer", "adverse", "hazards", "risk"]
    GOOD = ["benefit", "improvement", "advantage", "accuracy"]
    MORE = ["enhance", "higher", "exceed", "increase", "improve"]
    LESS = ["reduce", "decline", "fall"]
    
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

def cv(k=3):
    print '*'*70
    print 'Running...'
    score_training = 0

    s=0
    r=0
    f1=0
    
    q = np.zeros((3, 3))
    
    data = preprocessing.load(True)
    kf = KFold(data.shape[0], n_folds=k)
    for train_index, test_index in kf:
        training_data = data.iloc[train_index, :]
        test_data = data.iloc[test_index, :]
        print 'train_size: '+str(training_data.shape[0])
        print 'test_size: '+str(test_data.shape[0])
        
        data_x, ngram = training_ngram(training_data['sen'])
        data_y = training_data['lab']
        test_x = ngram.transform(test_data['sen']).toarray()
        test_y = test_data['lab']
        
        data_x = np.concatenate((data_x, training_change_phrase(training_data['sen'])), axis=1)
        test_x = np.concatenate((test_x, training_change_phrase(test_data['sen'])), axis=1)
        
        clf = svm.SVC(decision_function_shape='ovr', C=100)
        clf.fit(data_x, data_y)

        predict = clf.predict(test_x)
        
        s_ = metrics.precision_score(test_y, predict, average="weighted")
        r_ = metrics.recall_score(test_y, predict, average="weighted")
        f1_ = metrics.f1_score(test_y, predict, average="weighted")
        score_training += clf.score(data_x, data_y)
        s += s_
        r += r_
        f1 += f1_
        q[0,:] += eva(0, test_y, predict)
        print eva(0, test_y, predict)
        q[1,:] += eva(1, test_y, predict)
        q[2,:] += eva(2, test_y, predict)
    score_training /= len(kf)
    s /= 1.0*len(kf)
    r /= 1.0*len(kf)
    f1 /= 1.0*len(kf)
    q /= 1.0*len(kf)
    
    print '*'*70
    print 'Finish!'
    return s, r, f1, q

data_x, ngram = training_ngram(training_data['sen'])
data_y = training_data['lab']
test_x = ngram.transform(test_data['sen']).toarray()
test_y = test_data['lab']

data_x = np.concatenate((data_x, training_change_phrase(training_data['sen'])), axis=1)
#_, senti,senti_score1 = sentiwordnet.score(raw_training['sen'], raw_training['lab'])
#data_x = np.concatenate((data_x, np.array(senti).reshape((raw_training.shape[0],1))), axis=1)

test_x = np.concatenate((test_x, training_change_phrase(test_data['sen'])), axis=1)
#_, senti,senti_score1 = sentiwordnet.score(raw_test['sen'], raw_test['lab'])
#test_x = np.concatenate((test_x, np.array(senti).reshape((raw_test.shape[0],1))), axis=1)

clf = svm.SVC(decision_function_shape='ovr', C=100)
clf.fit(data_x, data_y)
predict = clf.predict(test_x)
s = metrics.precision_score(test_y, predict, average="weighted")
r = metrics.recall_score(test_y, predict, average="weighted")
f1 = metrics.f1_score(test_y, predict, average="weighted")
print f1

predict1 = clf.predict(data_x)
_, predict2,senti_score1 = sentiwordnet.score(raw_training['sen'], raw_training['lab'])
predict2 = map(float, predict2)
X = np.vstack((predict1, predict2)).T
X = np.array((predict1.tolist(), predict2)).T
clf1 = svm.SVC(decision_function_shape='ovr', C=0.1)
clf1.fit(X, training_data['lab'])
_, predict3, senti_score2 = sentiwordnet.score(raw_test['sen'], raw_test['lab'])
Test = np.vstack((predict.tolist(), predict3)).T

print 'score trainings: '+str(clf1.score(X, training_data['lab']))
print 'score test final: '+str(clf1.score(Test, test_data['lab']))
predict = clf1.predict(Test)
s1 = metrics.precision_score(test_data['lab'], predict, average="weighted")
r1 = metrics.recall_score(test_data['lab'], predict, average="weighted")
f11 = metrics.f1_score(test_data['lab'], predict, average="weighted")

def eva(lb, test_y, predict):
    test_y_ = test_y.copy()
    index = (test_y_==lb)
    test_y_[index]=1
    test_y_[~index]=0
    #print 'sum test='+str(sum(test_y_))
    predict_ = predict.copy()
    index = (predict_==lb)
    predict_[index]=1
    predict_[~index]=0
    #print 'predict ='+str(sum(predict_))
    s = metrics.precision_score(test_y_, predict_)
    r = metrics.recall_score(test_y_, predict_)
    f1 = metrics.f1_score(test_y_, predict_)
    return s, r, f1
