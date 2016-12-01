# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 08:53:11 2016

@author: ductr
"""

import pandas as pd
import preprocessing as pp

training_data, test_data, raw_training, raw_test, raw = pp.load()
new = pd.read_csv(base_directory + 'test_fail.csv')
new = new.dropna()
for i in new.index:
    new_id = new.loc[i, 'id']
    new_lab = new.loc[i,'decision']
    raw.loc[raw.id==new_id, 'lab'] = new_lab