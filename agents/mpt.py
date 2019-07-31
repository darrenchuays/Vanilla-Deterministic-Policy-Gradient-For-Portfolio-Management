# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 12:32:32 2019
Adapted from https://anaconda.org/mcg/portfolio/notebook
https://courses.csail.mit.edu/6.867/wiki/images/a/a7/Qp-cvxopt.pdf
@author: Darren
"""
import numpy as np
np.random.seed(7)
import pandas as pd
from matplotlib import pyplot as plt
import cvxopt
from cvxopt import blas, solvers

class MPT:
    def __init__(self):
        self.size = 0
    def forward(self, window_returns):
        solvers.options['show_progress'] = False
        cvxopt.solvers.options['abstol'] = cvxopt.solvers.options['reltol'] = cvxopt.solvers.options['feastol'] = 1e-8
        n = window_returns.shape[0]
        sigma = np.cov(window_returns)
        avg_ret = np.mean(window_returns, axis = 1)
        P = 2 * cvxopt.matrix(sigma)
        q = cvxopt.matrix(np.zeros(n))
        G = cvxopt.matrix(-np.eye(n,n))
        h = cvxopt.matrix(np.zeros(n))
        A = cvxopt.matrix(np.ones((1,n)))
        b = cvxopt.matrix(np.ones(1))
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)
        sol_x = np.array(solution['x'])[:,0]
        sol_x = np.concatenate((np.zeros(1), sol_x))
        sol_x *= (sol_x > 1e-5)
        min_risk_weight = sol_x / sol_x.sum()
        return min_risk_weight
