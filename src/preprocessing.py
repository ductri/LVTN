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

def load(is_one_set=False):

        data = pd.read_csv("F:\\code\\python\\lvtn\\final.csv", dtype={'sen':str})
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
        
        # Preprocessing
        data['sen'] = preprocessing(data['sen'])
        
        if not is_one_set:
            training_size = data.shape[0]*2/3
            
            training = data[0:training_size]
            test = data[training_size:]
            test.index = np.arange(test.shape[0])
            print 'training_size: '+str(training.shape[0])
            print 'test_size: '+str(test.shape[0])
            return training, test
        else:
            return data
            
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
    p=re.compile('\?|%|@|#|\^|\$|\.|,|;|:|/|"')
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
    #dataframe = pd.DataFrame(listwords)
    #dataframe.to_csv("F:\\code\\python\\lvtn\\src\\input.txt", sep="\n", index=None)
    dict_listwords = {}
    for word in listwords:
        dict_listwords[word] = word
    #os.system("java -jar F:\code\python\lvtn\src\metamap2.jar F:\code\python\lvtn\src\input.txt F:\code\python\lvtn\src\metamap_output.txt")
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