from functions import *
from tqdm import tqdm

'''
$Y= e + tau * T +  0.5 * Z$
   
$T = c + 0.2 * Z$
'''


def generate_data_confounder(tau, N, p, corr):
    nvar = p+1
    corr = 0.5  # correlation for multivariate normal
    allX = fn_generate_multnorm(N, corr, nvar)
    Z = allX[:, 1].reshape([N, 1])  # confounder
    T = fn_randomize_treatment(N)  # choose treated units
    e = np.random.normal(0, 1, [N, 1])
    Yab = tau * T + 0.5 * Z + e
    Tab = T + 0.2 * Z

    return (Yab, Tab, Z)


tau = 2
corr = 0.5
N = 1000
p = 3

Yab, Tab, Z = generate_data_confounder(tau, N, p, corr)
dt2 = pd.DataFrame(np.concatenate(
    [Yab, Tab, Z], axis=1), columns=['Yab', 'Tab', 'Z'])
dt2.to_csv('./data/2.csv')

# Not control confounder
estDict = {}
R = 1000
for N in [100, 1000]:
    tauhats = []
    sehats = []
    for r in tqdm(range(R)):
        Yab, Tab, Z = generate_data_confounder(tau, N, p, corr)
        covars = np.concatenate([Tab], axis=1)
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

# Control confounder

estDict = {}
R = 1000
for N in [100, 1000]:
    tauhats = []
    sehats = []
    for r in tqdm(range(R)):
        Yab, Tab, Z = generate_data_confounder(tau, N, p, corr)
        covars = np.concatenate([Tab, Z], axis=1)
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
