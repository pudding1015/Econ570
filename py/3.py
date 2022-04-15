from functions import *
from tqdm import tqdm

'''
Y = e + tau * T 
  
Z = c + 0.3 * T + 0.6 * Y
'''


def generate_data_select_bias(tau, N, p, corr):

    nvar = p+1  # 1 for selection bias
    corr = 0.5  # correlation for multivariate normal

    allX = fn_generate_multnorm(N, corr, nvar)

    T = fn_randomize_treatment(N)  # choose treated units
    e = np.random.normal(0, 1, [N, 1])
    c = np.random.normal(0, 1, [N, 1])
    Yab = e + tau * T
    Zab = c + 0.3 * T + 0.6 * Yab

    return (Yab, T, Zab)


tau = 0.5
corr = 0.5
N = 1000
p = 3
Yab, T, Zab = generate_data_select_bias(tau, N, p, corr)
dt3 = pd.DataFrame(np.concatenate(
    [Yab, T, Zab], axis=1), columns=['Yab', 'T', 'Zab'])
dt3.to_csv('./data/3.csv')

# Not control selection bias

estDict = {}
R = 1000
for N in [100, 1000]:
    tauhats = []
    sehats = []
    for r in tqdm(range(R)):
        Yab, T, Zab = generate_data_select_bias(tau, N, p, corr)
        Yt = Yab[np.where(T == 1)[0], :]
        Yc = Yab[np.where(T == 0)[0], :]
        tauhat, se_tauhat = fn_tauhat_means(Yt, Yc)
        tauhats = tauhats + [tauhat]
        sehats = sehats + [se_tauhat]

    estDict[N] = {
        'tauhat': np.array(tauhats).reshape([len(tauhats), 1]),
        'sehat': np.array(sehats).reshape([len(sehats), 1])
    }

tau0 = tau*np.ones([R, 1])
for N, results in estDict.items():
    (bias, rmse, size) = fn_bias_rmse_size(tau0, results['tauhat'],
                                           results['sehat'])
    print(f'N={N}: bias={bias}, RMSE={rmse}, size={size}')

# Control selection bias
estDict = {}
R = 1000
for N in [100, 1000]:
    tauhats = []
    sehats = []
    for r in tqdm(range(R)):
        Yab, T, Zab = generate_data_select_bias(tau, N, p, corr)
        covars = np.concatenate([T, Zab], axis=1)
        mod = sm.OLS(Yab, covars)
        res = mod.fit()
        tauhat = res.params[0]
        se_tauhat = res.HC1_se[0]
        tauhats = tauhats + [tauhat]
        sehats = sehats + [se_tauhat]

    estDict[N] = {
        'tauhat': np.array(tauhats).reshape([len(tauhats), 1]),
        'sehat': np.array(sehats).reshape([len(sehats), 1])
    }

tau0 = tau*np.ones([R, 1])
for N, results in estDict.items():
    (bias, rmse, size) = fn_bias_rmse_size(tau0, results['tauhat'],
                                           results['sehat'])
    print(f'N={N}: bias={bias}, RMSE={rmse}, size={size}')
