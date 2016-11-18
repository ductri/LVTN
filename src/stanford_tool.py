# -*- coding: utf-8 -*-
"""
Created on Sat Oct 22 16:42:57 2016

@author: ductr
"""

import os
from nltk.parse import stanford
import time
import preprocessing
import pandas as pd

training_data, test_data, raw_training, raw_test, raw = preprocessing.load()
os.environ['STANFORD_PARSER'] = 'F:\\code\\python\\lvtn\\stanford_nlp\\stanford-parser-full-2015-04-20\\stanford-parser-full-2015-04-20\\stanford-parser.jar'
os.environ['STANFORD_MODELS'] = 'F:\\code\\python\\lvtn\\stanford_nlp\\stanford-parser-full-2015-04-20\\stanford-parser-full-2015-04-20\\stanford-parser-3.5.2-models.jar'
java_path = "C:\\Program Files\\Java\\jre1.8.0_111\\bin\\java.exe"
os.environ['JAVAHOME'] = java_path
os.environ['CLASSPATH'] = 'F:\\code\\python\\lvtn\\stanford_nlp\\stanford-corenlp-full-2015-12-09\\stanford-corenlp-full-2015-12-09'

parser = stanford.StanfordNeuralDependencyParser()
output = 'F:\\code\\python\\lvtn\\stanford_nlp\\output'
index = 0
anchor = 221
for id, sen in zip(raw['id'][anchor:], raw['sen'][anchor:]):
    print '-'*50
    print 'Parsing sentences ' +str(index) 
    start = time.time()
    print 'Proccessing sentence with id ' + str(id)
    sentences = parser.raw_parse(sen)
    dependences = [list(parse.triples()) for parse in sentences][0]
    dependences = pd.DataFrame(dependences)
    
    lefts = zip(*dependences[0])
    dependences['left'] = lefts[0]
    dependences['left_POS'] = lefts[1]
    rights = zip(*dependences[2])
    dependences['right'] = rights[0]
    dependences['right_POS'] = rights[1]
    dependences = dependences.drop(dependences.columns[[0,2]], 1)

    
    dependences.to_csv(output+'\\'+str(id)+'.csv', index=None)
    index += 1
    time_consuming = (time.time()-start)/60
    print 'Done. Time consuming: '+ str(time_consuming) +'m'
    print 'Estimate time remain: '+str((raw['id'][anchor:].shape[0]-index)*time_consuming/60)+'h'
    print '-'*50