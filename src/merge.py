# -*- coding: utf-8 -*-
"""
Created on Sun Sep 11 16:17:48 2016

@author: ductr
"""

import pandas as pd
from sklearn.metrics import confusion_matrix
import numpy as np

linh = pd.read_csv('F:\\code\\python\\lvtn\\sentence_linh.csv')
submission = pd.read_csv('F:\\code\\python\\lvtn\\submission_new.csv')

submission = submission[['s_id','label']]
submission.columns = ['id','label']

final = pd.merge(linh, submission, on='id')
final.loc[final['re_label'].isnull(), 're_label'] = final[final['re_label'].isnull()]['label']
final = final[['id', 'content', 're_label']]
final.columns = ['id', 'sen', 'lab']
final = final.drop_duplicates(subset='id')

sentence_official = pd.read_csv('F:\code\python\lvtn\sentence_official.csv')
sentence_official.columns = ['id', 'sen', 'lab']
sentence_official = sentence_official[pd.notnull(sentence_official['lab'])]
sentence_official.loc[sentence_official['lab']=='POS', 'lab']=2
sentence_official.loc[sentence_official['lab']=='NEU', 'lab']=1
sentence_official.loc[sentence_official['lab']=='NEG', 'lab']=0
sentence_official['lab'] = sentence_official['lab'].astype('int64')

merge = pd.merge(sentence_official, final, on='id', how='outer')
merge.loc[merge['lab_x'].isnull(),'sen_x'] = merge[merge['lab_x'].isnull()]['sen_y']
merge.loc[merge['lab_y'].isnull(),'sen_y'] = merge[merge['lab_y'].isnull()]['sen_x']

merge.loc[merge['lab_x'].isnull(),'lab_x'] = merge[merge['lab_x'].isnull()]['lab_y']
merge.loc[merge['lab_y'].isnull(),'lab_y'] = merge[merge['lab_y'].isnull()]['lab_x']





#KAPPA
confusion_mat = confusion_matrix(merge['lab_x'], merge['lab_y'])

total_line = np.sum(confusion_mat, axis=0)
total_col = np.sum(confusion_mat, axis=1)
total= np.sum(total_line)
po = (confusion_mat[0][0] + confusion_mat[1][1] + confusion_mat[2][2])*1.0/total
pe = (total_line[0]*total_col[0] +total_line[1]*total_col[1] + total_line[2]*total_col[2])*1.0/total/total

kappa = []
kappa.append((po-pe)*1.0/(1-pe))

merge = merge[['id','sen_x','lab_x']]
merge.columns = ['id', 'sen', 'lab']
merge.to_csv('F:\\code\\python\\lvtn\\final.csv', index=None)