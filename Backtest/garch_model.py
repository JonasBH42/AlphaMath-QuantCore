import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model
from statsmodels.stats.diagnostic import acorr_ljungbox
import gc



# 1. Load your data and compute the volatility series
df = pd.read_csv("C:/Users/User/Desktop/Projects/Backtest/Csvs/NQ1!_MAIN_1D.csv")
df['volatility'] = abs(df['close'] - df['open']).dropna() * 0.1  # Adjusting volatility scale

# 2. Fit GARCH(1,1)
model = arch_model(df['volatility'], mean="ARX", lags=5, vol="GARCH", p=2, o=2, q=2, dist="skewt")
res   = model.fit(disp='off', update_freq=1)

# 3. Pull in-sample “forecast” ⇒ conditional volatility
df['vol_forecast'] = res.conditional_volatility.dropna()

print(df['vol_forecast'].dropna().head())

std_resid = res.std_resid.dropna()
lb1 = acorr_ljungbox(std_resid, lags=[10, 20], return_df=True)
lb2 = acorr_ljungbox(std_resid**2, lags=[10, 20], return_df=True)
print("Ljung-Box on resid:\n", lb1)
print("Ljung-Box on resid^2:\n", lb2)

print("GARCH(1,1) Model Summary:")
print(res.summary())
# 4. Plot
plt.figure(figsize=(10, 4))
plt.plot(df['volatility'],    label='Observed Volatility')
plt.plot(df['vol_forecast']*1.8,   label='GJR-GARCH(2, 2, 2) with AR mean', color='red')
plt.legend()
plt.title('Close-Open Volatility vs. GJR-GARCH(2, 2, 2) with AR mean')
plt.show()


gc.collect()