# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 17:32:48 2016

@author: ductr
"""

import sys
reload(sys)
sys.setdefaultencoding('utf8')
import preprocessing
import pandas as pd
import numpy as np

base_dir = 'F:/code/python/lvtn'
new = pd.read_csv(base_dir +'/new.csv')
new = new.dropna()
new.loc[new['lab']=='POS', 'lab']=2
new.loc[new['lab']=='NEU', 'lab']=1
new.loc[new['lab']=='NEG', 'lab']=0
new.to_csv(base_dir+'/new.csv', index=None)