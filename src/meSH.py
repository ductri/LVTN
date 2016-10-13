# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 11:55:16 2016

@author: Flynn
"""

import pandas as pd
from bs4 import BeautifulSoup
import requests
import re
from nltk.corpus import stopwords
import nltk
import os
import time

def preprocessing(corpus):
    # Lower text
    corpus = [sen.lower() for sen in corpus]
    corpus = remove_special_char(corpus)
    corpus = label_number(corpus)
    corpus = filter_stopwords(corpus)
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
    
def mesh(word):
    r = requests.post('https://www.nlm.nih.gov/cgi/mesh/2017/MB_cgi', data = {'term':word})
    soup = BeautifulSoup(r.text, 'html.parser') 
    temp = len(soup.find_all('title'))
    if (temp > 1):
        mheading = soup.find_all('title')[1].text
    else:
        mheading = word
    #time.sleep(1000)
    return mheading


src = "./../standard.csv"
src_data_raw = pd.read_csv(src, dtype={'sen':str})
src_data = preprocessing(src_data_raw['sen'])

words = [[w for w in sen.split(' ')] for sen in src_data]
listwords = []
for sen in words:
   for w in sen:
       listwords.append(w)

dict_listwords = {}

total_time = 0
speed = 0
index = 0
no_label = 0
index=3980
for word in listwords[3981:]:
    index += 1
    print '-_-'*20
    start = time.time()
    label = mesh(word)
    #try:
    if word in dict_listwords.keys():
        print 'Already have'
        continue
    else: dict_listwords[word] = label
    
    #except Exception as e:
        #print e
    if label!=word:
        print 'Word ' + str(index) +': ' + '"'+word+'"' + ' is replaced with '+'"'+label+'"'
        no_label += 1
    else: print 'Word ' + str(index) +': ' + '"'+word+'"' + 'is NOT replaced'
    
    duration = time.time()-start
    total_time += duration
    speed = total_time/index

    no_remaining = len(listwords) - index
    print 'Speed: '+str(speed)+'s/w'
    print 'Remaining words: ' + str(no_remaining)
    print 'Remaining time: '+str(speed*no_remaining/60)+ 'm'

data = pd.DataFrame({'key':dict_listwords.keys(), 'value':dict_listwords.values()})
data.to_csv('mesh.csv', index=None)