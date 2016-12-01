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
from nltk.stem.wordnet import WordNetLemmatizer

base_directory = "F:\\code\\python\\lvtn\\"
def preprocessing(corpus):
    # Lower text
    corpus = [sen.lower() for sen in corpus]
    corpus = remove_special_char(corpus)
    corpus = label_number(corpus)
    #corpus = filter_stopwords(corpus) #better result
    #corpus = metamaping(corpus)
    
    #corpus = stemming(corpus)
    corpus = lemmatization(corpus)
    corpus = stemming(corpus)
    #pos = pos_tagging(corpus)    
    #corpus_pos = stemming2(pos)
    #corpus = [[w[0]+w[1] for w in sen] for sen in corpus_pos]
    #corpus = [reduce(lambda x,y: x+' '+y, sen) for sen in corpus]
    
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
    #os.system("java -jar F:\code\python\lvtn\src\metamap2.jar F:\code\python\lvtn\src\input.txt F:\code\python\lvtn\src\metamap_output.txt")
    #output = pd.read_csv("mesh.csv")
    #output = pd.read_table("F:\code\python\lvtn\src\metamap_output.txt", sep=' ', header=None)
    output = pd.read_csv(base_directory+'src//metamap_output.csv')
    temp = output['0']
    temp = stemming(lemmatization(temp))
    output['0'] = temp
    for i in range(output.shape[0]):
        dict_listwords[output['0'][i]] = output['1'][i].upper()
    words = [[dict_listwords[w] for w in sen] for sen in words]
    result = [reduce(lambda x, y: x+' '+y, sen) for sen in words]
    return result

def pos_tagging(corpus):
    pos_tag = [nltk.pos_tag(nltk.word_tokenize(sen)) for sen in corpus]
    return pos_tag

def stemming(corpus):
    stemmer = SnowballStemmer("english")
    temp = [[stemmer.stem(w) for w in sen.split(' ')] for sen in corpus]
    result = [reduce(lambda x,y: x+' '+y, sen) for sen in temp]
    return result

def stemming2(pos):
    corpus_pos =[zip(*sen) for sen in pos]
    corpus_pos = zip(*corpus_pos)
    
    stemmer = SnowballStemmer("english")
    stem = [[stemmer.stem(w) for w in sen] for sen in corpus_pos[0]]
    corpus_pos[0] = stem
    corpus_pos = zip(corpus_pos[0], corpus_pos[1])    
    corpus_pos = [zip(sen[0], sen[1]) for sen in corpus_pos]
    return corpus_pos

def lemmatization(corpus):
    lmtzr = WordNetLemmatizer()
    temp = [[lmtzr.lemmatize(w) for w in sen.split(' ')] for sen in corpus]
    result = [reduce(lambda x,y: x+' '+y, sen) for sen in temp]
    return result
    
src= "F:\\code\\python\\lvtn\\relab_dataset2.csv" #standard_extend_fixed_ratio
src_data_raw = pd.read_csv(src, dtype={'sen':str})
src_data = preprocessing(src_data_raw['sen'])
#src_data = src_data_raw['sen']
#src= "F:\\code\\python\\lvtn\\so-cal.csv"
def load(is_one_set=False):
    
    data = src_data_raw.copy()
    data['sen'] =src_data
    
    data_size = data.shape[0]
    
    shuffle_index = np.arange(data_size)
    
    #TODO random here
    np.random.shuffle(shuffle_index)
    
    data = data.iloc[shuffle_index, :]
    data.index = np.arange(data_size)
    #print 'suffering done'    
    
    #print 'copy 2'    
    raw = src_data_raw.copy()
    raw = raw.iloc[shuffle_index, :]
    raw.index = np.arange(data_size)
    #print 'copy 2 done'    
    # Preprocessing
    #data['sen'] = preprocessing(data['sen'])
    ##print 'Preprocessing done!'
    if not is_one_set:
        training_size = data.shape[0]*2/3
        #print 'get data for training'
        training = data[0:training_size]
        #print 'get data for training done!'
        #print 'get data for testing'
        test = data[training_size:]
        #print 'get data for testing done!'
        test.index = np.arange(test.shape[0])
        
        #print 'get data raw'
        raw_training = raw[0:training_size]
        raw_test = raw[training_size:]
        raw_test.index = np.arange(test.shape[0])
        #print 'Done!'
        #print 'training_size: '+str(training.shape[0])
        #print 'test_size: '+str(test.shape[0])
        
        return training, test, raw_training, raw_test, raw
    else:
        return data, src_data_raw.copy()


    
