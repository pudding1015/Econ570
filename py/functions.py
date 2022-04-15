import pandas as pd
import numpy as np
import random
import statsmodels.api as sm
from tqdm import tqdm
import networkx as nx
import matplotlib.pyplot as plt
random.seed(6)


def fn_generate_cov(dim, corr):
    acc = []
    for i in range(dim):
        row = np.ones((1, dim)) * corr
        row[0][i] = 1
        acc.append(row)
    return np.concatenate(acc, axis=0)


def fn_generate_multnorm(nobs, corr, nvar):

    mu = np.zeros(nvar)
    std = (np.abs(np.random.normal(loc=1, scale=.5, size=(nvar, 1))))**(1/2)
    # generate random normal distribution
    acc = []
    for i in range(nvar):
        acc.append(np.reshape(np.random.normal(
            mu[i], std[i], nobs), (nobs, -1)))

    normvars = np.concatenate(acc, axis=1)

    cov = fn_generate_cov(nvar, corr)
    C = np.linalg.cholesky(cov)

    Y = np.transpose(np.dot(C, np.transpose(normvars)))

#     return (Y,np.round(np.corrcoef(Y,rowvar=False),2))
    return Y


def fn_randomize_treatment(N, p=0.5):
    treated = random.sample(range(N), round(N*p))
    return np.array([(1 if i in treated else 0) for i in range(N)]).reshape([N, 1])


def fn_randomize_treatment(N, p=0.5):
    treated = random.sample(range(N), round(N*p))
    return np.array([(1 if i in treated else 0) for i in range(N)]).reshape([N, 1])


def fn_tauhat_means(Yt, Yc):
    nt = len(Yt)
    nc = len(Yc)
    tauhat = np.mean(Yt)-np.mean(Yc)
    se_tauhat = (np.var(Yt, ddof=1)/nt+np.var(Yc, ddof=1)/nc)**(1/2)
    return (tauhat, se_tauhat)


def fn_bias_rmse_size(theta0, thetahat, se_thetahat, cval=1.96):
    """
    theta0 - true parameter value
    thetatahat - estimated parameter value
    se_thetahat - estiamted se of thetahat
    """
    b = thetahat - theta0
    bias = np.mean(b)
    rmse = np.sqrt(np.mean(b**2))
    tval = b/se_thetahat  # paramhat/se_paramhat H0: theta = 0
    size = np.mean(1*(np.abs(tval) > cval))
    # note size calculated at true parameter value
    return (bias, rmse, size)
