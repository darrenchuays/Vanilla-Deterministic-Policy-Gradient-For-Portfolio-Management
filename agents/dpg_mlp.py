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
import numpy as np
np.random.seed(7)
import matplotlib.pyplot as plt
import pandas as pd

class DPG_MLP(nn.Module):
    def __init__(self, M, L, N, hidden_size, predictor):
        super(DPG_MLP, self).__init__()
        self.predictor = predictor
        # Initial input shape
        self.M = M
        self.L = L
        self.N = N
        
        self.in_size = self.M*self.L*self.N
        self.n_hidden = hidden_size
        self.out_size = self.M
        
        self.drop_out = nn.Dropout(p=0.3)
        self.fc1 = nn.Linear(self.in_size, self.n_hidden)
        self.fc2 = nn.Linear(self.n_hidden, self.n_hidden)
        self.fc3 = nn.Linear(self.out_size, self.n_hidden)
        self.fc4 = nn.Linear(2*self.n_hidden, self.n_hidden)
        self.fc5 = nn.Linear(self.n_hidden, self.out_size)
            
   
    def forward(self, x, y):
        h1= torch.tanh(self.fc1(x))
        h2 = self.drop_out(torch.tanh(self.fc2(h1)))
        h3 = self.drop_out(torch.tanh(self.fc3(y)))
        h4 = torch.cat((h3,h2),dim=0)
        h5 = self.drop_out(torch.tanh(self.fc4(h4)))
        action_weights = torch.softmax(self.fc5(h5), dim=0)
        
        return action_weights
    