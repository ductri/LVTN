# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 09:43:40 2016

@author: ductr
"""

import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import nltk
import re
import os

def preprocessing(corpus):
    # Lower text
    corpus = [sen.lower() for sen in corpus]
    corpus = remove_special_char(corpus)
    corpus = label_number(corpus)
    corpus = filter_stopwords(corpus)
    corpus = metamaping(corpus)
    corpus = stemming(corpus)
    return corpus
def label_number(corpus):
    p=re.compile('([0-9]*([\.])?)[0-9]+')
    corpus = [p.sub('DIGIT', sen) for sen in corpus]
    return corpus

def remove_special_char(corpus):
    p=re.compile('\?|%|@|#|\^|\$|\.|,|;|:|/|"|\(|\)|\+|-|=')
    return [p.sub(' ',sen) for sen in corpus]
    
def filter_stopwords(text_corpus):
    liststopwords = stopwords.words('english')
    temp = [[w for w in nltk.word_tokenize(sen) if w.lower() not in 
    liststopwords] for sen in text_corpus]
    result = [reduce(lambda x,y: x+' '+y, sen) for sen in temp]
    return result
    
def metamaping(corpus):
    words = [[w for w in sen.split(' ')] for sen in corpus]
    listwords = []
    for sen in words:
        for w in sen:
            listwords.append(w)
    #1
    #dataframe = pd.DataFrame(listwords)
    #2
    #dataframe.to_csv("F:\\code\\python\\lvtn\\src\\input.txt", sep="\n", index=None)
    dict_listwords = {}
    for word in listwords:
        dict_listwords[word] = word
    #3
    #os.system("java -jar F:\code\python\lvtn\src\metamap4.jar F:\code\python\lvtn\src\input.txt F:\code\python\lvtn\src\metamap_output.txt")
    output = pd.read_csv("F:\\code\\python\\lvtn\\src\\metamap_output.txt", sep=" ", header=None)
    for i in range(output.shape[0]):
        dict_listwords[output[0][i]] = output[1][i]
    words = [[dict_listwords[w] for w in sen] for sen in words]
    result = [reduce(lambda x, y: x+' '+y, sen) for sen in words]
    return result
    
def stemming(corpus):
    stemmer = SnowballStemmer("english")
    temp = [[stemmer.stem(w) for w in sen.split(' ')] for sen in corpus]
    result = [reduce(lambda x,y: x+' '+y, sen) for sen in temp]
    return result
    
    
src= "F:\\code\\python\\lvtn\\standard.csv"
src_data_raw = pd.read_csv(src, dtype={'sen':str})
src_data = preprocessing(src_data_raw['sen'])

#src= "F:\\code\\python\\lvtn\\so-cal.csv"
def load(is_one_set=False):
    print 'begin loading'
    data = src_data_raw.copy()
    data['sen'] =src_data
    print 'copy 1 done'
    data_size = data.shape[0]
    
    # Just classify 0 and the others
    #data.loc[data['lab']!=0,'lab']=1
    #data.loc[data['lab']!=2,'lab']=1
    #data.loc[data['lab']==2,'lab']=0
    # Shuffle dataframe
    shuffle_index = np.arange(data_size)
    
    np.random.shuffle(shuffle_index)
    data = data.iloc[shuffle_index, :]
    data.index = np.arange(data_size)
    print 'suffering done'    
    
    print 'copy 2'    
    raw = src_data_raw.copy()
    print 'copy 2 done'    
    # Preprocessing
    #data['sen'] = preprocessing(data['sen'])
    #print 'Preprocessing done!'
    if not is_one_set:
        training_size = data.shape[0]*2/3
        print 'get data for training'
        training = data[0:training_size]
        print 'get data for training done!'
        print 'get data for testing'
        test = data[training_size:]
        print 'get data for testing done!'
        test.index = np.arange(test.shape[0])
        
        print 'get data raw'
        raw_training = raw[0:training_size]
        raw_test = raw[training_size:]
        raw_test.index = np.arange(test.shape[0])
        print 'Done!'
        print 'training_size: '+str(training.shape[0])
        print 'test_size: '+str(test.shape[0])
        
        return training, test, raw_training, raw_test, raw
    else:
        return data, src_data_raw.copy()
def load_raw():
    data = pd.read_csv(src, dtype={'sen':str})
    data_size = data.shape[0]
    
    # Just classify 0 and the others
    #data.loc[data['lab']!=0,'lab']=1
    #data.loc[data['lab']!=2,'lab']=1
    #data.loc[data['lab']==2,'lab']=0
    # Shuffle dataframe
    shuffle_index = np.arange(data_size)
    np.random.shuffle(shuffle_index)
    data = data.iloc[shuffle_index, :]
    data.index = np.arange(data_size)
    training_size = data.shape[0]*2/3
            
    training = data[0:training_size]
    test = data[training_size:]
    test.index = np.arange(test.shape[0])
    print 'training_size: '+str(training.shape[0])
    print 'test_size: '+str(test.shape[0])
    return training, test
            

    
