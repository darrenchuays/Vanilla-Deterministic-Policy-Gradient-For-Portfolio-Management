# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 09:42:26 2019

@author: Darren
"""


import torch
torch.manual_seed(7)
torch.cuda.manual_seed(7)

class UCRP:
    def __init__(self):
        self.size = 0
        
    def forward(self, x, y):
        self.size = y.shape[0]-1
        action = torch.ones(self.size)/self.size
        action = torch.cat((torch.zeros(1), action))
        return action