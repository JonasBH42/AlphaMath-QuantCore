# df = pd.read_csv('C:/Users/User/Desktop/AlphaMath-QuantCore/Backtest/Csvs/NQ1!_MAIN_1H.csv', parse_dates=True, index_col=0)

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.gofplots import qqplot
from scipy.stats import genpareto
from hurst import compute_Hc
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('C:/Users/User/Desktop/AlphaMath-QuantCore/Backtest/Csvs/NQ1!_MAIN_1M.csv', parse_dates=True, index_col=0)
df.drop(columns=["rtype", "publisher_id", "instrument_id", "symbol"], inplace=True)
df = df.iloc[int(len(df) * 0.5):]

length = 50

df['20_sma'] = df['close'].ewm(span=length, adjust=False).mean()
df['r']      = (df['close'] - df['20_sma']).dropna()
print("Results for length =", length)

# 1. ADF test on r_t
adf_result = adfuller(df['r'].dropna())
print('ADF stat:',    adf_result[0])
print('p-value:',     adf_result[1])
print('crit values:', adf_result[4])

# 2. OU fit and κ
r  = df['r'].dropna()
r0 = r.values[:-1]
r1 = r.values[1:]
ou_mod = sm.OLS(r1, r0).fit()
phi   = ou_mod.params[0]
kappa = -np.log(phi)
print('φ =', phi, 'κ =', kappa)

# 3. Half-life
t_half = np.log(2) / kappa
print('Half-life ≈', t_half)

# 4. QQ-plot vs Normal
qqplot(df['r'].dropna(), line='s')
plt.title('QQ-plot of r_t vs Normal')
plt.show()

# 5. Tail-fit with Generalized Pareto (POT method)
threshold = np.percentile(df['r'].dropna(), 95)
excesses  = df['r'][df['r'] > threshold] - threshold
c, loc, scale = genpareto.fit(excesses)
print('GPD params (c, loc, scale):', (c, loc, scale))

# 6. Rolling Hurst exponent (safe)
def safe_hurst(series):
    if np.all(series == series[0]):
        return np.nan
    try:
        H, _, _ = compute_Hc(series, kind='price', simplified=True)
        return H
    except FloatingPointError:
        return np.nan

window = 250
hurst_vals = [np.nan]*len(df)
for i in range(window, len(df)):
    hurst_vals[i] = safe_hurst(df['r'].iloc[i-window:i].values)
df['hurst'] = hurst_vals

# 7. 2-state HMM on returns (suppress underflow errors)
returns = df['r'].dropna().values.reshape(-1,1)
scaler  = StandardScaler()
returns_scaled = scaler.fit_transform(returns)

hmm = GaussianHMM(
    n_components=2,
    covariance_type='diag',
    n_iter=1000,
    tol=1e-4,
    verbose=True
)

# suppress underflow warnings during fit
with np.errstate(under='ignore'):
    hmm.fit(returns_scaled)

states = hmm.predict(returns_scaled)
df.loc[df['r'].dropna().index, 'hmm_state'] = states

# 8. Monte-Carlo ES calculation
n_sim        = 100_000
bootstrapped = np.random.choice(r, size=n_sim)
var95        = np.percentile(bootstrapped, 5)
es95         = bootstrapped[bootstrapped <= var95].mean()
print('5% VaR =', var95, 'Expected Shortfall =', es95)




        
