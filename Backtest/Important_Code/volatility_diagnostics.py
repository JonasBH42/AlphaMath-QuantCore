import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.diagnostic import het_arch, acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.stattools import jarque_bera
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm
from arch import arch_model

# 1) Load data & compute high–low volatility
CSV_PATH = "C:/Users/User/Desktop/Projects/Backtest/Csvs/NQ1!_MAIN_1D.csv"
df = pd.read_csv(CSV_PATH)
df["volatility"] = abs(df["close"] - df["open"]) 
vol = df["volatility"].dropna()* 0.1
arma = ARIMA(vol, order=(2, 0, 1)).fit()
resid = arma.resid.dropna()

# 2) ARCH-LM test (nlags → q for GARCH(q,·))
lm_stat, lm_pvalue, f_stat, f_pvalue = het_arch(df["volatility"], nlags=12)
print(f"\nARCH-LM test (12 lags): LM p-value = {lm_pvalue:.4g}")

# 3) ACF & PACF of squared vols to choose p, q
# plot_acf(vol**2, lags=20, title="ACF of vol²")
# plot_pacf(vol**2, lags=20, title="PACF of vol²")
# plt.show()

# 4) Preliminary GARCH(1,1) fit
res = arch_model(vol, mean="ARX", lags=5, vol="GARCH", p=2, o=2, q=2, dist="skewt").fit(
    disp="off"
)
print("\nPreliminary GARCH(1,1) Results:\n", res.summary())

# 5) Sign‐bias test for leverage/asymmetry
std_resid = res.std_resid.dropna()


neg = (std_resid < 0).astype(int).shift(2).dropna()
y = (std_resid**2).loc[neg.index]
X = sm.add_constant(neg)
sb = sm.OLS(y, X).fit()
print("neg vals: ", neg.value_counts())
print(sb.summary())

# 6) Jarque–Bera test for heavy tails on std. residuals
jb_stat, jb_pvalue, skew, kurt = jarque_bera(std_resid)
print(
    f"\nJarque–Bera on std_resid: p-value={jb_pvalue:.4g}, skew={skew:.2f}, kurtosis={kurt:.2f}"
)

# 7) Ljung–Box on raw & squared std. residuals
lb_raw = acorr_ljungbox(std_resid, lags=[10, 20], return_df=True)
lb_sq = acorr_ljungbox(std_resid**2, lags=[10, 20], return_df=True)
print("\nLjung–Box on std_resid:\n", lb_raw)
print("\nLjung–Box on std_resid^2:\n", lb_sq)

_, p_arch, _, _ = het_arch(std_resid, nlags=12)
print("ARCH-LM std_resid p:", p_arch)


# 8) ARMA on the raw vol series (mean‐process check)
arma = ARIMA(vol, order=(1, 0, 1)).fit()
print("\nARMA(1,0,1) on vol:\n", arma.summary())

# 9) Summary guidance
print(
    """
==> Interpretation Guide <==
• ARCH-LM p<0.05 → need a GARCH(q,·) with q ≥ number of lags until p>0.05.
• ACF of vol²: slowly decaying → large β or higher q.
• PACF of vol²: spikes at lag k → consider p ≥ k.
• Sign-bias: significant → use asymmetric model (GJR or EGARCH).
• JB p<0.05 or high kurtosis → switch to dist='t' or 'skewt'.
• Ljung–Box on std_resid/raw & squared: want all p>0.05 before accepting a specification.
• ARMA significance: if ARMA terms are significant, include them via mean='ARX', lags=…
"""
)
