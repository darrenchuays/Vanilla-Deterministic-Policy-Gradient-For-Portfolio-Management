# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 13:16:24 2019

@author: Darren
"""

import torch
torch.manual_seed(7)
torch.cuda.manual_seed(7)
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.distributions as td
from argparse import ArgumentParser
import json
import pandas as pd
import numpy as np
np.random.seed(7)
import math
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from agents.dpg_mlp import DPG_MLP
from agents.mpt import MPT
from agents.UCRP import UCRP
import datetime
import os
import seaborn as sns
sns.set_style("darkgrid")

cost = 0.0025
rc_factor = 1
err = 1e-9
PATH_prefix = ''

def max_drawdown(wealth):
    drawdown = wealth.detach().clone().numpy()
    roll_max = np.maximum.accumulate(drawdown)
    drawdown = drawdown / (roll_max + err) - 1
    MDD = drawdown.min()
    return MDD*100
    
def sharpe_ratio(daily_returns):
    exp_return = daily_returns.mean()
    std_dev = daily_returns.std()
    SR = np.sqrt(252)*exp_return/std_dev
    return SR, exp_return*100, std_dev*100

def consolidate_info(pd, label, SR, exp_return, std_dev, mdd, cum_growth):
    pd.loc[label, 'Sharpe Ratio'] = SR
    pd.loc[label, 'Expected Daily Return %'] = exp_return
    pd.loc[label, 'Std Daily Returns % '] = std_dev
    pd.loc[label, 'MDD % '] = mdd
    pd.loc[label, 'Cumulative Growth '] = cum_growth
    
def train_agent(agent, env, epoch, codes):
    num_traj = 5
    total_loss = torch.tensor(0)
    codes = ['Cash'] + codes
    all_cum_daily_changes = []
    mpt = MPT()
    for i in range(num_traj):
        G_t, y_t, price_t, terminal = env.reset()
        cum_daily_changes = torch.zeros(y_t.shape[0])
        w1_t = torch.zeros(G_t.shape[0])
        w1_t[0] = 1
        G_t = torch.Tensor(G_t)
        optimizer = torch.optim.Adam(agent.parameters())
        loss = torch.tensor(0)
        t = 0
        while terminal:
            w2_t = agent.forward(G_t.reshape(-1), w1_t)
            mu = cost * (torch.abs(w2_t[1:] - w1_t[1:])).sum()
            G_t2, y_t2, price_t2, terminal = env.step()
            window_returns = price_t2[1:]/price_t[1:]
            min_risk_w = mpt.forward(window_returns)
            min_risk_w = torch.Tensor(min_risk_w)
            y_t2 = torch.Tensor(y_t2)
            reward = torch.dot(w2_t, y_t2) * (1-mu)
            reward = torch.log(reward + err)
            w1_t2 = (w2_t *y_t2) / (torch.dot(w2_t, y_t2) + err)
            cum_daily_changes = cum_daily_changes + torch.log(y_t2.clone())         #This was added to check verify that the algorithm only maximises the log returns
            loss = loss + reward
            w1_t = w1_t2
            G_t = torch.Tensor(G_t2)
            price_t = price_t2
            t = t + 1
        loss = loss/t
        cum_daily_changes = cum_daily_changes/t
        total_loss = total_loss + loss
        all_cum_daily_changes.append(cum_daily_changes)
    all_cum_daily_changes = torch.stack(all_cum_daily_changes).mean(dim=0)
    best_cum_daily_changes = codes[int(all_cum_daily_changes.max(0)[1])]
    total_loss = total_loss/num_traj 
    print(total_loss)
    growth = (torch.exp(-total_loss.clone()) - 1)*100                       #This is for the average daily return due to random start date
    print('Total steps per epoch:', t)
    print('Final Training Growth:', growth)
    print('Final weight allocation:', w1_t)
    print('This is the cumulative daily changes:', all_cum_daily_changes)
    print('This is the best cumulative daily changes:', best_cum_daily_changes)
    agent.zero_grad()
    total_loss.backward()
    optimizer.step()
    return t, growth
       
def backtest(pg,env, codes, mode, market, predictor):
    global PATH_prefix

    agents = []
    agents.append(pg)
    ucrp = UCRP()
    agents.append(ucrp)
    mpt = MPT()
    agents.append(mpt)
    strategy_labels = [predictor, 'UCRP', 'MinVar']
    all_growth = pd.DataFrame()
    final_info = pd.DataFrame(index=strategy_labels)
    for i, strategy in enumerate(agents):
        G_t, y_t, price_t, terminal = env.reset()
        total_w1 = []
        w1_t = torch.zeros(G_t.shape[0])
        w1_t[0] = 1
        total_w1.append(w1_t)
        G_t = torch.Tensor(G_t)
        total_reward = torch.tensor(0)
        daily_returns = []
        cumulative_growth = []
        t = 0
        while terminal:
            if strategy_labels[i] == 'MinVar':
                window_returns = price_t[1:, 1:]/price_t[1:, :-1]
                w2_t = strategy.forward(window_returns)
                w2_t = torch.Tensor(w2_t)            
            else:
                w2_t = strategy.forward(G_t.reshape(-1), w1_t)
            
            mu = cost * (torch.abs(w2_t[1:] - w1_t[1:])).sum()
            G_t2, y_t2, price_t2, terminal = env.step()
            y_t2 = torch.Tensor(y_t2)
            reward = torch.dot(w2_t, y_t2) * (1-mu)
            reward = -torch.log(reward)
            w1_t2 = (w2_t *y_t2) / torch.dot(w2_t, y_t2)
            total_reward = total_reward + reward
            daily_returns.append(torch.exp(-reward)-1)
            cumulative_growth.append(torch.exp(-total_reward))
            w1_t = w1_t2
            G_t = torch.Tensor(G_t2)
            price_t = price_t2
            total_w1.append(w1_t)
            t = t + 1
        SR, exp_return, std_dev = sharpe_ratio(torch.stack(daily_returns))
        mdd = max_drawdown(torch.stack(cumulative_growth))
        print("Calculating Sharpe Ratio with average daily return of:", exp_return.item(), ' %', 'and std of:', std_dev.item(), ' %')
        print('Sharp ratio for '+ strategy_labels[i] + ':', SR)
        print('MDD for '+ strategy_labels[i] + ':', mdd)
        print('Total steps per epoch:', t)
        print('Final Test Growth for ' + strategy_labels[i] + ':', cumulative_growth[-1])
        print('Final weight allocation for ' + strategy_labels[i] + ':', w1_t)
        consolidate_info(final_info, strategy_labels[i], SR.item(), exp_return.item(), std_dev.item(), mdd.item(), cumulative_growth[-1].item())
        np.savetxt(r'' + PATH_prefix + strategy_labels[i] + '_weights_' + mode, torch.stack(total_w1).data.numpy(), delimiter=",")
        all_growth[strategy_labels[i]] = torch.stack(cumulative_growth).data.numpy()
        plt.plot(cumulative_growth, label = strategy_labels[i])
        
    all_growth.to_csv(r'' + PATH_prefix + 'all_growth_' + mode + '_rc_' + str(rc_factor) + '_' + predictor + '_' + market + '_hs_' + str(pg.n_hidden) + '_back_test.csv')
    final_info.to_csv(r'' + PATH_prefix + 'final_info_' + mode + '_rc_' + str(rc_factor) + '_' + predictor + '_' + market + '_hs_' + str(pg.n_hidden) + '_back_test.csv')
    plt.legend()  
    plt.savefig(r'' + PATH_prefix + mode + '_rc_' + str(rc_factor) + '_' + predictor + '_' + market + '_hs_' + str(pg.n_hidden) + '_back_test.png') 
    plt.show()

def parse_config(config):
    num_assets = config["session"]["num_assets"]
    start_date = config["session"]["start_date"]
    end_date = config["session"]["end_date"]
    features = config["session"]["features"]
    agent_config = config["session"]["agents"]
    market = config["session"]["market_types"]
    actor, framework, window_length, hidden_size = agent_config
    epochs = int(config["session"]["epochs"])

    print("*--------------------Training Status-------------------*")
    print("Date from",start_date,' to ',end_date)
    print('Features:',features)
    print("Market Type:",market)
    print("Predictor:",actor,"  Framework:", framework,"  Window_length:",window_length, " Hidden_size:", hidden_size)
    print("Epochs:",epochs)

    


    return num_assets, start_date, end_date, features, agent_config, market, actor, framework, window_length, hidden_size, epochs

def session(config): 
    global PATH_prefix
    from data.environment import Environment
    num_assets, start_date, end_date, features, agent_config, market, predictor, framework, window_length, hidden_size, epochs = parse_config(config)
    env = Environment()
    PATH_prefix = "result/PG/"

    if not os.path.exists(PATH_prefix):
        os.makedirs(PATH_prefix)
        train_start_date, train_end_date, test_start_date, test_end_date, assets_tickers = env.preprocess_data(start_date, end_date, num_assets, market, window_length, features)
        print("Assets_tickers:", assets_tickers)
        print('Training Time Period:', train_start_date, '   ', train_end_date)
        print('Testing Time Period:', test_start_date, '   ', test_end_date)
        with open(PATH_prefix + 'config.json', 'w') as f:
            json.dump({"train_start_date": train_start_date.strftime('%Y-%m-%d'),
                       "train_end_date": train_end_date.strftime('%Y-%m-%d'),
                       "test_start_date": test_start_date.strftime('%Y-%m-%d'),
                       "test_end_date": test_end_date.strftime('%Y-%m-%d'), "assets_tickers": assets_tickers}, f)
        print("finish writing config")
        
    with open("result/PG/config.json", 'r') as f:
        dict_data = json.load(f)
        print("Config loaded successfully!")
    train_start_date, train_end_date, assets_tickers = datetime.datetime.strptime(dict_data['train_start_date'], '%Y-%m-%d'), datetime.datetime.strptime(dict_data['train_end_date'], '%Y-%m-%d'), dict_data['assets_tickers']
    env.get_data(train_start_date, train_end_date, features, window_length, market, assets_tickers, 'Train')

    print("*-----------------Loading PG Agent---------------------*")
    if predictor == 'MLP':
        agent = DPG_MLP(len(assets_tickers) + 1, int(window_length), len(features), hidden_size, predictor)
        print('Actor: ' + predictor + 'network')
        print(agent.eval())
    elif predictor == 'LSTM':
        agent = DPG_LSTM(len(assets_tickers) + 1, int(window_length), len(features), hidden_size, predictor)
        print('Actor: ' + predictor + 'network')
        print(agent.eval())
        
    print("Training with {:d}".format(epochs))
    growth_all_epoch = []
    for epoch in range(epochs):
        print("Now we are at epoch", epoch)
        total_train_days, growth = train_agent(agent, env, epoch, assets_tickers)
        growth_all_epoch.append(growth)
    plt.plot(growth_all_epoch)
    plt.savefig(r'' + PATH_prefix + 'Training_growth_rc_' + str(rc_factor) + '_' + predictor + '_' + market + '_hs_' + str(hidden_size) +'.png') 
    plt.show()
        
    # Back Test
    print("Running back test now.....")
    test_start_date,test_end_date,assets_tickers = datetime.datetime.strptime(dict_data['test_start_date'],'%Y-%m-%d'), datetime.datetime.strptime(dict_data['test_end_date'],'%Y-%m-%d'), dict_data['assets_tickers']
    env.get_data(train_start_date, train_end_date, features, window_length, market, assets_tickers, 'Test')
    backtest(agent, env, assets_tickers, 'Train_Set', market, predictor)
    env.get_data(test_start_date, test_end_date, features, window_length, market, assets_tickers, 'Test')
    backtest(agent, env, assets_tickers, 'Test_Set', market, predictor)
    del agent


def main():
    with open('config.json') as f:
        config=json.load(f)
        session(config) 

if __name__=="__main__":
    main()