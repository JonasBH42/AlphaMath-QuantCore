import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model
from statsmodels.stats.diagnostic import acorr_ljungbox
import gc



# 1. Load your df and compute the volatility series
df = pd.read_csv("C:/Users/User/Desktop/AlphaMath-QuantCore/Backtest/Csvs/NQ1!_MAIN_1D.csv")
df['volatility'] = abs(df['close'] - df['open']).dropna() * 0.1  # Adjusting volatility scale

# 2. Fit GARCH(1,1)
model = arch_model(df['volatility'], mean="ARX", lags=5, vol="GARCH", p=2, o=2, q=2, dist="skewt")
res   = model.fit(disp='off', update_freq=1)

# 3. Pull in-sample “forecast” ⇒ conditional volatility
df['vol_forecast'] = res.conditional_volatility.dropna()
df['vol_forecast'] = df['vol_forecast'] * 2.8
# True Range (TR)
df["tr"] = np.maximum.reduce([
    df["high"] - df["low"],
    (df["high"] - df["close"].shift(1)).abs(),
    (df["low"] - df["close"].shift(1)).abs()
])

# Average True Range (ATR) over a 14-day window
df["ATR"] = df["tr"].rolling(window=1, min_periods=1).mean()


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
plt.plot(df['vol_forecast'],   label='GJR-GARCH(2, 2, 2) with AR mean', color='red')
plt.legend()
plt.title('Close-Open Volatility vs. GJR-GARCH(2, 2, 2) with AR mean')
plt.show()


gc.collect()