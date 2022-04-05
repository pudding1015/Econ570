import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn import metrics
# Y = a + b * X + 0.3 * Z
# X = 0.2 * Z + e

a = 1.2
b = 2.1
n = 100
sample_size = 100 # 1000

slope_estimates_a = []
slope_estimates_b = []
bias_estimates_a = []
bias_estimates_b = []
X_list = []
Z_list = []
W_list = []
Y_list = []

for mc_replication in range(sample_size):
    e = np.random.normal(0,1,n)
    Z = np.random.uniform(0, 1, n)
    X = 0.2 * Z + e
    Y = a + b * X +0.3 * Z
    
    # a. Control for the variable in between the path from cause to effect    
    mod_a = sm.OLS(Y, sm.add_constant(X))
    res_a = mod_a.fit()
    slope_estimates_a = slope_estimates_a + [res_a.params[1]]
    bias_estimates_a = bias_estimates_a + [res_a.params[0]]
    
    # b. Do not control for the variable in between the path from cause to effect
    mod_b = sm.OLS(Y, sm.add_constant(Z))
    res_b = mod_b.fit()
    slope_estimates_b = slope_estimates_b + [res_b.params[1]]
    bias_estimates_b = bias_estimates_b + [res_b.params[0]]

    X_list.extend(X)
    Y_list.extend(Y)

df = pd.DataFrame({'X':X,'Y':Y})
df.to_csv("./3.csv", index=None) 

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