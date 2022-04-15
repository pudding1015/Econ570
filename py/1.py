from functions import *
from tqdm import tqdm

'''
Y = tau * T + c * Z + d * W + e
'''


def fn_generate_data(tau, N, p, p0, corr, conf=True, flagX=False):
    """
    p0(int): number of covariates with nonzero coefficients
    """
    nvar = p+2  # 1 confounder and variable for randomizing treatment
    corr = 0.5  # correlation for multivariate normal

    if conf == False:
        conf_mult = 0  # remove confounder from outcome

    allX = fn_generate_multnorm(N, corr, nvar)
    W0 = allX[:, 0].reshape([N, 1])  # variable for RDD assignment
    C = allX[:, 1].reshape([N, 1])  # confounder
    X = allX[:, 2:]  # observed covariates

    T = fn_randomize_treatment(N)  # choose treated units
    err = np.random.normal(0, 1, [N, 1])
    beta0 = np.random.normal(5, 5, [p, 1])

    beta0[p0:p] = 0  # sparse model
    Yab = tau*T+X@beta0+conf_mult*0.5*C+err
    if flagX == False:
        return (Yab, T)
    else:
        return (Yab, T, X)


tau = 0.5
corr = 0.5
conf = False
p = 3
p0 = 2
flagX = 1
N = 1000
Yab, T, X = fn_generate_data(tau, N, p, p0, corr, conf, flagX)
dt1 = pd.DataFrame(np.concatenate([Yab, T, X], axis=1), columns=[
                   'Yab', 'T', 'X1', 'X2', 'X3'])
dt1.to_csv('./data/1.csv')

# Not control covariates

estDict = {}
R = 1000
for N in [100, 1000]:
    tauhats = []
    sehats = []
    for r in tqdm(range(R)):
        Yab, T, X = fn_generate_data(tau, N, p, p0, corr, conf, flagX)
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

# Control covariates
estDict = {}
R = 1000
for N in [100, 1000]:
    tauhats = []
    sehats = []
    for r in tqdm(range(R)):
        Yab, T, X = fn_generate_data(tau, N, p, p0, corr, conf, flagX)
        X_obs = X[:, :p0]
        covars = np.concatenate([T, X_obs], axis=1)
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
