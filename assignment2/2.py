import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn import metrics

# Y = a + b * X , Z = c + d * X

a = 1.5
b = 1.2
c = 2.1
d = 2.5
n = 100
sample_size = 100 # 1000

X_list = []
Z_list = []
W_list = []
Ya_list = []
Yb_list = []


slope_estimates_a = []
slope_estimates_b = []
bias_estimates_a = []
bias_estimates_b = []

for mc_replication in range(sample_size):
    X = np.random.uniform(0, 1, n)

    # a. fail to control for the confounder
    Ya = a + b * X
    Z = c + d * X 
    mod_a = sm.OLS(Ya, sm.add_constant(Z))
    res_a = mod_a.fit()
    slope_estimates_a = slope_estimates_a + [res_a.params[1]]
    bias_estimates_a = bias_estimates_a + [res_a.params[0]]
    
    # b. control for the confounder
    Yb = a + b * X 
    mod_b = sm.OLS(Yb, sm.add_constant(X))
    res_b = mod_b.fit()
    slope_estimates_b = slope_estimates_b + [res_b.params[1]]
    bias_estimates_b = bias_estimates_b + [res_b.params[0]]
    
    X_list.extend(X)
    Z_list.extend(Z)
    Ya_list.extend(Ya)
    Yb_list.extend(Yb)

df = pd.DataFrame({'X':X,'Z':Z,'Ya':Ya,'Yb':Yb})
df.to_csv("./data/2.csv", index=None) 

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