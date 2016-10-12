# -*- coding: utf-8 -*-
"""
Created on Mon Oct 03 23:39:37 2016

@author: ductr
"""

import preprocessing
import requests
from HTMLParser import HTMLParser
from bs4 import BeautifulSoup
import pandas as pd
import time

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
    #os.system("java -jar F:\code\python\lvtn\src\metamap3.jar F:\code\python\lvtn\src\input.txt F:\code\python\lvtn\src\metamap_output.txt")
    output = pd.read_csv("F:\\code\\python\\lvtn\\src\\metamap_output.txt", sep=" ", header=None)
    for i in range(output.shape[0]):
        dict_listwords[output[0][i]] = output[1][i]
    words = [[dict_listwords[w] for w in sen] for sen in words]
    result = [reduce(lambda x, y: x+' '+y, sen) for sen in words]
    return result

start = time.time()
print("start")
src= "F:\\code\\python\\lvtn\\standard.csv"
data = pd.read_csv(src, dtype={'sen':str})
data['sen'] = metamaping(data['sen'])

data['socal']=0

for i in range(data.shape[0]):
    r = requests.post('http://www.cs.sfu.ca/~sentimen/socal/SO_Web.cgi', 
                  data = {'user_input':data['sen'][i]})
    soup = BeautifulSoup(r.text, 'html.parser')
    data.loc[i, 'socal'] = float(soup.find_all('p')[1].text[1:-1])
    print("Dont "+str(i))
data.to_csv("F:\\code\\python\\lvtn\\so-cal.csv")
end = time.time()
print("finish")
duration = end-start
print(duration)



    