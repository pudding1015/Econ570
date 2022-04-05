import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn import metrics

# Y = a + b * X + c * Z + d * W

a = 0.5
c = 1.2
d = 2.1
n = 100
sample_size = 100 # 1000

slope_estimates_a = []
slope_estimates_b = []
bias_estimates_a = []
bias_estimates_b = []
X_list = []
Z_list = []
W_list = []
Ya_list = []
Yb_list = []

for mc_replication in range(sample_size):
    b = np.random.rand(1)
    X = np.random.uniform(0, 1, n)
    Z = np.random.uniform(0, 1, n)
    W = np.random.uniform(0, 1, n)
    Ya = a + b * X + c * Z + d * W  # a.You do not control for any covariates
    Yb = a + b * X                  # b.You control for all the covariates that affect the outcome
    
    X_list.extend(X)
    Z_list.extend(Z)
    W_list.extend(W)
    Ya_list.extend(Ya)
    Yb_list.extend(Yb)

    mod_a = sm.OLS(Ya, sm.add_constant(X))
    res_a = mod_a.fit()
    slope_estimates_a = slope_estimates_a + [res_a.params[1]]
    bias_estimates_a = bias_estimates_a + [res_a.params[0]]

    mod_b = sm.OLS(Yb, sm.add_constant(X))
    res_b = mod_b.fit()
    slope_estimates_b = slope_estimates_b + [res_b.params[1]]
    bias_estimates_b = bias_estimates_b + [res_b.params[0]]

df = pd.DataFrame({'X':X,'Z':Z,'W':W,'Ya':Ya,'Yb':Yb})
df.to_csv("./data/1.csv", index=None)

# a
print('slope_estimate:',np.mean(slope_estimates_a))
print('bias_estimate:',np.mean(bias_estimates_a))
b0 = b*np.ones([100,1])
temp = slope_estimates_a - b0
bias = np.mean(temp)
rmse = np.sqrt(np.mean(temp**2))
print('RMSE:',rmse,'bias:',bias)

# b
print('slope_estimate:',np.mean(slope_estimates_b))
print('bias_estimate:',np.mean(bias_estimates_b))
b0 = b*np.ones([100,1])
temp = slope_estimates_b - b0
bias = np.mean(temp)
rmse = np.sqrt(np.mean(temp**2))
print('RMSE:',rmse,'bias:',bias)