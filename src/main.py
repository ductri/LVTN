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


import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
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
import re
import json
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


base_directory = "F:\\code\\python\\lvtn\\"

# 0:NEG, 1:NEU, 2:POS
training_data, test_data, raw_training, raw_test, raw = preprocessing.load()
socal = pd.read_csv(base_directory + "so-cal.csv")
pca = LinearDiscriminantAnalysis()
pca.fit(socal['socal'].reshape(-1,1), socal['lab'])
socal['y'] = pca.transform(socal['socal'].reshape(-1,1))
predict2 = []
for s in socal['socal']:
    if s<-0.5:
        predict2.append(0)
    elif s<0.5:
        predict2.append(1)
    else: predict2.append(2)
        
def training_ngram(corpus):
    vectorizer = CountVectorizer(min_df=3, decode_error="ignore", analyzer="word", 
                                        lowercase=True, binary=True, ngram_range=(1,2),
                                        stop_words='english') #stop_words = 'enghlish' is the best
    data_array = vectorizer.fit_transform(corpus).toarray()
#    vectorizer = TfidfVectorizer(min_df=3, decode_error="ignore", analyzer="word", 
#                                        lowercase=True, binary=True, ngram_range=(1,2),
#                                        stop_words='english')
#                                        
#    data_array = vectorizer.fit_transform(corpus).toarray()
    return data_array, vectorizer
def training_change_phrase(corpus):
    BAD = ["suffer", "adverse", "hazards", "risk", "death", "insufficient",
           "infection", "recurrence", "restlessness", "mortality", "hazard",
           "chronic", "pain", "negative", "severity","complication","risk","adverse","mortality","morbidity","death","fatal","danger","no benefit","discourage","short-term risk","long-term risk","damage","little information","not been well studies","ineffective","suffer","depression","acute","sore","outpatient","disabling","diabetes","difficulties","dysfunction","distorted","poorer","unable","prolonged","irritation","disruptive","pathological","mutations","disease","infection","harms","difficulty","weakened","inactive","stressors","hypertension","adverse","insomnia","relapsing","malignant","suffer","exacerbate","dryness","fever","overestimate","constipation","deposition","colic","tension","hazards","diarrhoea","weakness","irritability","insidious","distress","weak","cancer","emergency","risk","block","unsatisfactory ","blinding","nausea","traumatic","wound","intention","loses","intensive","relapse","recurrent","extension","die","cancers","malaise","crying","toxic","injury","confounding","complaints","misuse","insignificant","poisoning","anoxic","amputation","death","nightmares","deteriorate","fatal","injuries","fatigue","invasive","suicide","chronic","relapsed","disturbances","confusion","died","fluctuating","severities","delusions","compulsions","conflict","trauma","cried","impair","severe","tremor","weaker","illness","inpatients","worry","rebound","worse","reversible","dizziness","attacks","pointless","disorders","dyskinesia","risks","fatty","negative","conflicting","upset","fishy","hard","harm","bleeding","inflammatory","hampered","underpowered","obstruction","headache","problem","bleeds","panic","loss","odds","retardation","dysfunctional","render","difficult","drowsiness","lack","suicidal","obsessions","impaired","cough","severity","suffering","violent","strokes","virus","stroke","flatulence","fibrates","blind ","burning ","faintness","suffered","threatening","misdiagnosing","bitter","excessive","diabetics","malfunction","abnormal","deterioration","bad","confounded","sadness","mortality","disturbance","agitated","attack","infections","negativistic","deaths","poor","wrong","worsening","adversely","insufficient","scarring","headaches","disability","overdose ","serious","delayed","discomfort","sweating","morbidity","nerve","parkinson","toxicity","nervous","pain","stress","weakens","incorrect","disorder","worsened","malformations","blinded","rigidity","prolong","adversity","abuse","lacked","dyspepsia","sads ","onset","failure","inadequate","sensitivity","impairment","dementia","harmful"]
    GOOD = ["benefit", "improvement", "advantage", "accuracy", "great",
            "effective", "support", "potential", "superior", "mild", "achieved",
           "Supplementation", "beneficial", "positive","benefit","beneficial","improve","advantage","resolve","good","fantastic","relief","superior","efficacious","effective","improve effectiveness","importance of protecting","significant advantage","significant therapeutic advantage","may be effective","effective approach","simple and effective","simple and effective treatment","safe","well tolerated","well-tolerated","useful","maybe useful","illustrate the benefits","significant improvement","significantly improve","clinically worthwhile","worthwhile","recover rapid","satisfactory outcome","satisfactory","similarly effective","supports","approve","more effective","high efficacy","cured","vitality","relaxing","benefit","tolerability","improvement","right","effective","stable","best","better","pleasurable","relaxation","favour","beneficial","safety","prevents","successful","satisfaction","significant","superior","contributions","reliability","robust","tolerated","improving","survival","favourable","reliable","recovered","judiciously","consciousness","efficacy","prevented","satisfied","prevent","advantage","encouraging","tolerance","success","significance","improved","improves","improve","improvements"]
    MORE = ["enhance", "higher", "exceed", "increase", "improve", "somewhat",
            "quite", "very", "higher", "more", "augments", "highest","enhance","augment","increase","amplify","raise","boost","add to","higher","exceed","rise","go up","surpass","more","additional","extra","added","greater","positive","high","prolonged","prolong","increase","enhance","elevation","higher","exceed","enhancement","peaked","more","excess"]
    LESS = ["reduce", "decline", "fall", "less", "little", "slightly", "only", 
            "mildly", "smaller", "lower", "reduction", "drop","fewer","slump",
            "fall","down","pummel","less","lower","low","decrease","reduce",
            "decline","descend","collapse","fail","subside","lesser","poorer",
            "Worse","smaller","negative","prevent","reduced","prevents","below",
            "lower","decrease","fall","low","reduce","decline","less","little",
            "mild","drop","fewer"]
    
    BAD = preprocessing.stemming(preprocessing.lemmatization(BAD))
    GOOD = preprocessing.stemming(preprocessing.lemmatization(GOOD))
    MORE = preprocessing.stemming(preprocessing.lemmatization(MORE))
    LESS = preprocessing.stemming(preprocessing.lemmatization(LESS))
    #print 'len change phrase: '+str(len(BAD)+len(GOOD)+len(MORE)+len(LESS))
    def sen2vec(sen):
        words = sen.split(' ')
        vecs = [0,0,0,0] #MORE GOOD, MORE BAD, LESS GOOD, LESS BAD
        for i in range(len(words)):
            if words[i] in MORE:
                #print 'more='+words[i]
                for k in range(i, len(words)):
                    if words[k] in GOOD:
                        vecs[0] = 1
                        break
                    if words[k] in BAD:
                        #print 'bad='+words[k]
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

def training_change_phrase1(corpus):
    BAD = ["suffer", "adverse", "hazards", "risk", "death", "insufficient",
           "infection", "recurrence", "restlessness", "mortality", "hazard",
           "chronic", "pain", "negative", "severity","complication","risk","adverse","mortality","morbidity","death","fatal","danger","no benefit","discourage","short-term risk","long-term risk","damage","little information","not been well studies","ineffective","suffer","depression","acute","sore","outpatient","disabling","diabetes","difficulties","dysfunction","distorted","poorer","unable","prolonged","irritation","disruptive","pathological","mutations","disease","infection","harms","difficulty","weakened","inactive","stressors","hypertension","adverse","insomnia","relapsing","malignant","suffer","exacerbate","dryness","fever","overestimate","constipation","deposition","colic","tension","hazards","diarrhoea","weakness","irritability","insidious","distress","weak","cancer","emergency","risk","block","unsatisfactory ","blinding","nausea","traumatic","wound","intention","loses","intensive","relapse","recurrent","extension","die","cancers","malaise","crying","toxic","injury","confounding","complaints","misuse","insignificant","poisoning","anoxic","amputation","death","nightmares","deteriorate","fatal","injuries","fatigue","invasive","suicide","chronic","relapsed","disturbances","confusion","died","fluctuating","severities","delusions","compulsions","conflict","trauma","cried","impair","severe","tremor","weaker","illness","inpatients","worry","rebound","worse","reversible","dizziness","attacks","pointless","disorders","dyskinesia","risks","fatty","negative","conflicting","upset","fishy","hard","harm","bleeding","inflammatory","hampered","underpowered","obstruction","headache","problem","bleeds","panic","loss","odds","retardation","dysfunctional","render","difficult","drowsiness","lack","suicidal","obsessions","impaired","cough","severity","suffering","violent","strokes","virus","stroke","flatulence","fibrates","blind ","burning ","faintness","suffered","threatening","misdiagnosing","bitter","excessive","diabetics","malfunction","abnormal","deterioration","bad","confounded","sadness","mortality","disturbance","agitated","attack","infections","negativistic","deaths","poor","wrong","worsening","adversely","insufficient","scarring","headaches","disability","overdose ","serious","delayed","discomfort","sweating","morbidity","nerve","parkinson","toxicity","nervous","pain","stress","weakens","incorrect","disorder","worsened","malformations","blinded","rigidity","prolong","adversity","abuse","lacked","dyspepsia","sads ","onset","failure","inadequate","sensitivity","impairment","dementia","harmful"]
    GOOD = ["benefit", "improvement", "advantage", "accuracy", "great",
            "effective", "support", "potential", "superior", "mild", "achieved",
           "Supplementation", "beneficial", "positive","benefit","beneficial","improve","advantage","resolve","good","fantastic","relief","superior","efficacious","effective","improve effectiveness","importance of protecting","significant advantage","significant therapeutic advantage","may be effective","effective approach","simple and effective","simple and effective treatment","safe","well tolerated","well-tolerated","useful","maybe useful","illustrate the benefits","significant improvement","significantly improve","clinically worthwhile","worthwhile","recover rapid","satisfactory outcome","satisfactory","similarly effective","supports","approve","more effective","high efficacy","cured","vitality","relaxing","benefit","tolerability","improvement","right","effective","stable","best","better","pleasurable","relaxation","favour","beneficial","safety","prevents","successful","satisfaction","significant","superior","contributions","reliability","robust","tolerated","improving","survival","favourable","reliable","recovered","judiciously","consciousness","efficacy","prevented","satisfied","prevent","advantage","encouraging","tolerance","success","significance","improved","improves","improve","improvements"]
    MORE = ["enhance", "higher", "exceed", "increase", "improve", "somewhat",
            "quite", "very", "higher", "more", "augments", "highest","enhance","augment","increase","amplify","raise","boost","add to","higher","exceed","rise","go up","surpass","more","additional","extra","added","greater","positive","high","prolonged","prolong","increase","enhance","elevation","higher","exceed","enhancement","peaked","more","excess"]
    LESS = ["reduce", "decline", "fall", "less", "little", "slightly", "only", 
            "mildly", "smaller", "lower", "reduction", "drop","fewer","slump",
            "fall","down","pummel","less","lower","low","decrease","reduce",
            "decline","descend","collapse","fail","subside","lesser","poorer",
            "Worse","smaller","negative","prevent","reduced","prevents","below",
            "lower","decrease","fall","low","reduce","decline","less","little",
            "mild","drop","fewer"]
    
    BAD = preprocessing.stemming(preprocessing.lemmatization(BAD))
    GOOD = preprocessing.stemming(preprocessing.lemmatization(GOOD))
    MORE = preprocessing.stemming(preprocessing.lemmatization(MORE))
    LESS = preprocessing.stemming(preprocessing.lemmatization(LESS))
    #print 'len change phrase: '+str(len(BAD)+len(GOOD)+len(MORE)+len(LESS))
    def sen2vec(sen):
        words = sen.split(' ')
        vecs = [0,0,0,0] #MORE GOOD, MORE BAD, LESS GOOD, LESS BAD
        windows_size = 4
        for i in range(len(words)):
            if words[i] in MORE:                
                for k in range(i, min(len(words),i+windows_size)):
                    if words[k] in GOOD:
                        vecs[0] = 1
                        break
                    if words[k] in BAD:
                        #print 'bad='+words[k]
                        vecs[1] = 1
                        break
                for k in range(i, len(words)):
                    words[k] = words[k] + '_MORE'
            elif words[i] in LESS:
                for k in range(i, min(len(words),i+windows_size)):
                    if words[k] in GOOD:
                        vecs[2] = 1
                        break
                    if words[k] in BAD:
                        vecs[3] = 1
                        break
                for k in range(i, len(words)):
                    words[k] = words[k] + '_LESS'
        sen = reduce(lambda x,y:x+' '+y,words)
        return vecs, sen
    result = [sen2vec(sen) for sen in corpus]
    result = zip(*result)
    return result
    
def training_negation(training_data, test_data):
    src = 'F:\code\python\lvtn\MetamapNegation\output.json'
    metamap = pd.read_json(src)
    #training_data, test_data, raw_training, raw_test, raw = preprocessing.load()
    training_data = pd.merge(training_data, metamap, how='inner', on='id') #TODO
    step1a(training_data)
    test_data = pd.merge(test_data, metamap, how='inner', on='id') #TODO
    step1a(test_data)
    return training_data, test_data
    
def step1(data):
    for i in range(data.shape[0]):
  
        sen = data.iloc[i,1]
        neg = data.iloc[i,3]
  
        if len(neg)==0:
            continue
        elif neg[0]['negex'].find(' no ')!=-1:
            list_neg = neg[0]['negex'].split(' ')[1:]
            list_neg = preprocessing.lemmatization(list_neg)
            list_neg = preprocessing.stemming(list_neg)
            for n in list_neg:
                index = sen.find(n)
                if index==-1:
                    raise
                else:
                    new_sen = sen[:index+len(n)]+'_NEG'+sen[index+len(n):]
                    data.iloc[i,1] = new_sen
                    print '-'*10
                    
def step1a(data):
    for i in range(data.shape[0]):
        sen = data.iloc[i,1]
        neg = data.iloc[i,3]
      
        if len(neg)==0:
            continue
        else:
            for neg_item in neg:
                data.iloc[i,1] = sen.replace(neg_item['negex'], 'NEGATION')



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
    
    #training_data, test_data = training_negation(training_data, test_data)
    
    #training_data['sen'] = preprocessing.metamaping(training_data['sen'])
    #test_data['sen'] = preprocessing.metamaping(test_data['sen'])
    
    
    
    data_x, ngram = training_ngram(training_data.sen)
    data_x = normalize(data_x)
    data_y = training_data['lab']*1.0
    test_x = ngram.transform(test_data.sen).toarray()
    test_x = normalize(test_x)
    test_y = test_data['lab']*1.0
    
    change_phrase_data_x = training_change_phrase(training_data['sen'])
    change_phrase_test_x = training_change_phrase(test_data['sen'])
    data_x = np.concatenate((data_x, change_phrase_data_x), axis=1)
    test_x = np.concatenate((test_x, change_phrase_test_x), axis=1)
    
    
    
    
##    #SOCAL
    
    
    temp = pd.merge(training_data, socal, on='id', how='left')
    data_x = np.concatenate((data_x, np.array(temp['y']).reshape((temp.shape[0],1))), axis=1)

    temp = pd.merge(test_data, socal, on='id', how='left')
    test_x = np.concatenate((test_x, np.array(temp['y']).reshape((temp.shape[0],1))), axis=1)
    
    #10
    clf = svm.SVC(decision_function_shape='ovr', C=c, kernel='rbf', 
                  class_weight='balanced')
    #clf = svm.NuSVC(nu=c, decision_function_shape='ovr')
    #0.4->0.5
    #clf = BernoulliNB(alpha=c)
    
    clf.fit(data_x, data_y)
    predict = clf.predict(test_x)
    
    a = metrics.accuracy_score(test_y, predict)
    s = metrics.precision_score(test_y, predict, average="weighted")
    r = metrics.recall_score(test_y, predict, average="weighted")
    f1 = metrics.f1_score(test_y, predict, average="weighted")
    
    f1_all = metrics.f1_score(test_y, predict, average=None)
    
    #print f1
    return predict, s, r, f1, f1_all, a

def cv(k, c):

    s=0
    r=0
    f1=0
    f1_all = np.zeros(3)
    a = 0
    data, data_raw = preprocessing.load(True)
    kf = KFold(data.shape[0], n_folds=k)
    for train_index, test_index in kf:
        training_data = data.iloc[train_index, :]
        raw_training = data_raw.iloc[train_index, :]
        test_data = data.iloc[test_index, :]
        raw_test = data_raw.iloc[test_index, :]
        #print 'train_size: '+str(training_data.shape[0])
        #print 'test_size: '+str(test_data.shape[0])
        
        predict, s_, r_, f1_,f1_all_, a_ = run(training_data, test_data,raw_training, raw_test,  c)

        s += s_
        r += r_
        f1 += f1_
        f1_all += f1_all_
        a += a_
  
    s /= 1.0*len(kf)
    r /= 1.0*len(kf)
    f1 /= 1.0*len(kf)
    f1_all /= len(kf)
    a /= len(kf)

    return s, r, f1, f1_all, a
    
def test(n=50, c=17, k=10):
    f1 = 0
    f1_all = 0
    a = 0
    for i in range(0,n):
        _,_,f1_, f1_all_, a_ = cv(k, c)
        f1+= f1_
        f1_all += f1_all_
        a += a_
        
    return f1/n, f1_all/n, a
a=[]
def findc(title=''):
    start = time.time()
    score = []
    score0 = []
    score1 = []
    score2 = []
    a = []
    c = np.arange(10, 20, 1)
    index = 0
    for i in c:
        print '-'*30
        print(str(index)+'/'+str(len(c)))
        print('c='+str(i))
        print '-'*30
        index = index + 1
        result = test(n=30, c=i, k=5)
        score.append(result[0])
        score0.append(result[1][0])
        score1.append(result[1][1])
        score2.append(result[1][2])
        a.append(result[2])
    plt.figure()
    l1, l2, l3, l4 = plt.plot(c, score, 'b-', c, score0, 'g--', c, score1, 'r--', c, score2, 'b--')
    plt.title(title)
    plt.xlabel('c')
    plt.ylabel('score f1')
    plt.legend([l1, l2, l3, l4], ['Average', 'Negative', 'Neutral', 'Positive'], loc=4, bbox_to_anchor=(1, 1))

    end = time.time()
    print('time: ' + str(end-start))
    print('Score max: '+ str(max(score)) + ' with c='+str(c[np.argmax(score)]))
