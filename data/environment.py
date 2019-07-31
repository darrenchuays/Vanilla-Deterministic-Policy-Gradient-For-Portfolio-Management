# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 13:16:24 2019

@author: Darren
"""

import numpy as np
np.random.seed(7)
import pandas as pd
from datetime import datetime
import time
import random
random.seed(7)
eps=10e-8

class Environment:
    def __init__(self):
        self.cost = 0.0025

    def preprocess_data(self, start_date, end_date, num_assets, market, window_length, features):
        #Read data from processed csv files
        self.data = pd.read_csv(r'./data/'+market+'.csv',index_col=0,parse_dates=True,dtype=object)
        self.data["code"]=self.data["code"].astype(str)
        self.data[features] = self.data[features].astype(float)
        self.date_set = pd.date_range(start_date,end_date)
        
        
        #Sample a random number of asset to invest
        codes = random.sample(set(self.data["code"]), num_assets)
        
        #Data-preprocessing for each asset
        data2 = self.data.loc[self.data["code"].isin(codes)]        
        preprocessed_data = []
        for asset in codes:
            asset_data = data2[data2["code"]==asset].reindex(self.date_set).sort_index()    #Collect data, expand to the full days (including weekends,PH) then sort them
            asset_data = asset_data.resample('D').mean()                                    #To drop last 3 columns
            asset_data['close'] = asset_data['close'].fillna(method='pad')                  #To pad weekends of close with previous day close
            asset_data = asset_data.fillna(method='bfill',axis=1)
            asset_data = asset_data.fillna(method='ffill',axis=1)                           #bfill and ffill to pad forward and backward in the axis=1 direction with closed price
            asset_data['code'] = asset
            preprocessed_data.append(asset_data)
            
        preprocessed_data = pd.concat(preprocessed_data)
        preprocessed_data.to_csv(r'./data/' + 'processed_' + market + '.csv')
            
        date_set = set(preprocessed_data.loc[preprocessed_data['code']==codes[0]].index)    #we use the first one since we assume data is preprocessed so that all assets have the same time frame
        self.date_set = list(date_set)
        self.date_set.sort()
        
        #Set the train and test date proper
        train_start_time = self.date_set[0]
        train_end_time = self.date_set[int(len(self.date_set) / 6) * 5 - 1]
        test_start_time = self.date_set[int(len(self.date_set) / 6) * 5 - int(window_length)] #we minus off window so that we start off the next day after the end of train set
        test_end_time = self.date_set[-1]

        return train_start_time,train_end_time,test_start_time,test_end_time,codes

    def get_data(self, start_time, end_time, features, window_length, market, codes, mode):
        #Read data from processed csv files
        self.mode = mode
        self.codes = codes
        self.data = pd.read_csv(r'./data/' + 'processed_' + market + '.csv', index_col=0, parse_dates=True, dtype=object)
        self.data["code"] = self.data["code"].astype(str)
        self.data[features] = self.data[features].astype(float)
        self.data = self.data[start_time.strftime("%Y-%m-%d"):end_time.strftime("%Y-%m-%d")]
        data = self.data

        #Initialize parameters
        self.M = len(codes)+1                                                           #Total asset size including cash
        self.N = len(features)                                                          #Number of features, e.g. 4 if using OHLC
        self.L = int(window_length)                                                     #Look_back window including current time step
        self.date_set = pd.date_range(start_time,end_time)                              #Set the date based on inputted start_time and end_time

        #Data-preprocessing for each asset
        asset_dict = dict()
        for asset in codes:
            asset_data = data[data["code"]==asset].sort_index()                         #Collect data, expand to the full days (including weekends,PH) then sort them
            asset_dict[str(asset)] = asset_data
                       
        #Pack the data into states use for environment 
        self.ext_states = []
        self.y_all_history = []
        self.P_close_all_history = []
        t = self.L
        self.length = len(self.date_set)
        print('Total days:', self.length)
        
        while t <= self.length:  
            ext_state=[]
            
            #Start:This is for cash
            y_all = np.ones(1)                                                                  
            P_close_all = np.ones(self.L)  #absolute close price                                             
            V_close_all = np.ones(self.L)  #relative close price wrt to current time step
            if 'high' in features:
                V_high_all = np.ones(self.L)
            if 'low' in features:
                V_low_all = np.ones(self.L)
            if 'open' in features:
                V_open_all = np.ones(self.L)
            #End
            
            for asset in codes:
                asset_data = asset_dict[str(asset)]
                P_close = asset_data.ix[t - self.L:t, 'close']
                V_close = asset_data.ix[t - self.L:t, 'close']/asset_data.ix[t-1, 'close']
                if t==self.L and self.mode!='Train':
                    print('This is start date:',asset_data.ix[t])
                V_close_all = np.vstack((V_close_all, V_close))
                P_close_all = np.vstack((P_close_all, P_close))
                
                if 'high' in features:
                    V_high = asset_data.ix[t - self.L:t, 'high']/asset_data.ix[t-1, 'close']
                    V_high_all = np.vstack((V_high_all, V_high))
                    
                if 'low' in features:
                    V_low = asset_data.ix[t - self.L:t, 'low']/asset_data.ix[t-1, 'close']
                    V_low_all = np.vstack((V_low_all, V_low))
                    
                if 'open' in features:
                    V_open = asset_data.ix[t - self.L:t, 'open']/asset_data.ix[t-1, 'close']
                    V_open_all = np.vstack((V_open_all, V_open))
                    
                y = asset_data.ix[t-1,'close']/asset_data.ix[t-2,'close']
                y_all = np.vstack((y_all, y))
            
            ext_state.append(V_close_all)
            
            if 'high' in features:
                ext_state.append(V_high_all)
            if 'low' in features:
                ext_state.append(V_low_all)
            if 'open' in features:
                ext_state.append(V_open_all)
                
            ext_state = np.stack(ext_state, axis=2)
            ext_state = ext_state.reshape(self.M, self.L, self.N)  #Reshape just in case
            self.ext_states.append(ext_state)
            self.y_all_history.append(y_all.reshape(-1))
            self.P_close_all_history.append(P_close_all)
            t = t+1
            
        self.reset()

    def step(self):
        self.t = self.t + 1
        not_done = 0
        if self.t < self.end-1:
            not_done = 1
        return self.ext_states[self.t], self.y_all_history[self.t], self.P_close_all_history[self.t], not_done
                

    def reset(self):
        self.t = 0
        self.end = len(self.ext_states)
        #For Random Start and End
        #if self.mode == 'Train':
            #self.t = random.randint(0, len(self.ext_states)-2)
            #self.end = random.randint(self.t + 1, len(self.ext_states))
        return self.ext_states[self.t], self.y_all_history[self.t], self.P_close_all_history[self.t], 1

    def get_codes(self):
        return self.codes
        