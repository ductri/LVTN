# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 16:53:35 2016

@author: ductr
"""
import nltk
from nltk.corpus import sentiwordnet as swn
from sklearn.metrics import confusion_matrix
#ADJ, ADJ_SAT, ADV, NOUN, VERB = 'a', 's', 'r', 'n', 'v'


def convert(word_tuple):
    if word_tuple[1] == 'VERB':
        return word_tuple[0]+'.v'
    elif word_tuple[1] == 'ADJ':
        return word_tuple[0]+'.a'
    elif word_tuple[1] == 'ADV':
        return word_tuple[0]+'.r'
    elif word_tuple[1] == 'NOUN':
        return word_tuple[0]+'.n'
    else: return ""

def score_word_with_tag(word):
    if word=='':
        return 0
    #sentis = swn.senti_synset(word)
    #scores = map(lambda x: x.pos_score() - x.neg_score(), sentis)
    try:
        senti = swn.senti_synset(word+'.1')
        #print senti
        score = senti.pos_score() - senti.neg_score()
    except nltk.corpus.reader.wordnet.WordNetError:
        score = 0
    return score
    
def score_sen(sen):
    text = nltk.word_tokenize(sen)
    tagged_words = nltk.pos_tag(text, tagset='universal')
    #print 'tagged_words:' + str(tagged_words)
    words_with_tag = map(lambda x: convert(x), tagged_words)
    #print 'words_with_tag:' + str(words_with_tag)
    score = map(lambda x: score_word_with_tag(x), words_with_tag)
    #print score
    return sum(score)*1.0/len(text)
    
    
def score(corpus, lab):
    list_score = map(lambda x: score_sen(x)+0.1, corpus)
    
    def convert(x):
        if x>0.4:
            return 2 
        elif x < -0.4:
            return 0 
        else: return 1
    ave = sum(list_score)*1.0/len(list_score)
    scale = max(map(abs,list_score))
    predict = map(lambda x: (x-ave+1)/scale, list_score)
    #predict = map(convert, list_score)
    #mat = confusion_matrix(predict, lab)
    #s = (mat[0][0]+mat[1][1]+mat[2][2])*1.0/sum(sum(mat))
    #print((mat[0][0]+mat[1][1]+mat[2][2])*1.0/sum(sum(mat)))
    return list_score
def predict(corpus):
    return map(score_sen, corpus)