# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 08:31:50 2019

@author: Darren
"""

import os
import pandas as pd
import numpy as np

list_of_files = os.listdir()
all_data = []
used_columns = ['open', 'high', 'low', 'close', 'code']
for each_file in list_of_files:
    if each_file.endswith('.csv'):
        temp = pd.read_csv(each_file, index_col = 'Date')
        temp.replace('', np.nan)
        tickers = each_file.split(".")[0]
        temp['code'] = tickers
        temp = temp.rename(index=str, columns={"Open": "open", "High": "high", "Low": "low" , "Close": "close"})
        print(temp.tail(10))
        temp= temp.fillna(method='ffill',axis=0)
        print(temp.tail(10))
        all_data.append(temp[used_columns])

all_data = pd.concat(all_data)
all_data.to_csv(r'../SG.csv')
        
