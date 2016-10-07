# -*- coding: utf-8 -*-
"""
Created on Thu Oct 06 15:28:45 2016

@author: ductr
"""

training_data, test_data, raw_training, raw_test, raw = preprocessing.load()
data = pd.concat((training_data, test_data))
data.index = range(data.shape[0])
change_phrase = pd.DataFrame(training_change_phrase(data['sen']))
myfilter = []
for i in range(change_phrase.shape[0]):
    myfilter.append(change_phrase.iloc[i][0] or 
                    change_phrase.iloc[i][1] or 
                    change_phrase.iloc[i][2] or 
                    change_phrase.iloc[i][3])
myfilter = np.array(myfilter)
sub = change_phrase[myfilter==1]
sizesub = sub.shape[0]
sub['id'] = range(sizesub)
x1=map(lambda x, y: x or y, sub[0], sub[2])
x2=map(lambda x, y: x or y, sub[1], sub[3])
x1 = pd.DataFrame({'x1':x1, 'id':range(sub.shape[0])})
x2 = pd.DataFrame({'x2':x2, 'id':range(sub.shape[0])})
sub = pd.merge(sub, x1, on='id')
sub = pd.merge(sub, x2, on='id')
y = data.loc[sub.index, 'lab']
y = pd.DataFrame({'y':y, 'id':range(sub.shape[0])})
sub = pd.merge(sub, y, on='id')