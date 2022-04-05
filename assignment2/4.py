import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn import metrics

# Y = 0 + 0 * X + e

a = 0
b = 0
n = 100
sample_size = 100 # 1000

slope_estimates = []
bias_estimates = []

X_list = []
Ya_list = []
Yb_list = []
for mc_replication in range(sample_size):
    X = np.random.uniform(0, 1, n)
    e = np.random.normal(0,1,n)
    Ya = a + b * X + e
    mod = sm.OLS(Ya, sm.add_constant(X))
    res = mod.fit()
    slope_estimates = slope_estimates + [res.params[1]]
    bias_estimates = bias_estimates + [res.params[0]]
    X_list.extend(X)
    Ya_list.extend(Ya)

df = pd.DataFrame({'X':X,'Ya':Ya})
df.to_csv("./data/4.csv", index=None) 

print('slope_estimate:',np.mean(slope_estimates))
print('bias_estimate:',np.mean(bias_estimates))
sum = 0
for i in range(len(slope_estimates)):
    sum+=(slope_estimates[i]-b)**2
RMSE = np.sqrt(sum/100)
print('RMSE_slope:',RMSE)
