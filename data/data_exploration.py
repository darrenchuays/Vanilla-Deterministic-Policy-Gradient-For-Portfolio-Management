# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 10:36:11 2019

@author: Darren
"""

import pandas as pd
import matplotlib.pyplot as plt
import random
import numpy as np

all_data = pd.read_csv('CN.csv', index_col = 'Date')
codes = list(set(all_data['code']))
#codes = random.sample(set(all_data["code"]), 5)

for each_code in codes:
    temp = all_data[all_data['code']==each_code]['close']
    temp = temp.reset_index()
    temp = temp['close']/(temp['close'].iloc[-1])
    plt.plot(temp.index, temp, label = each_code)
    plt.xticks(rotation='vertical')

plt.legend()
plt.show()