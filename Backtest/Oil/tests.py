import pandas as pd
import numpy as np
from scipy.stats import boxcox, boxcox_normmax
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import warnings
from tqdm import tqdm
warnings.filterwarnings("ignore")

# 1. Load data
path = "C:/Users/User/Desktop/AlphaMath-QuantCore/Backtest/Csvs/Crude_Oil_1D.csv"
df = pd.read_csv(path, parse_dates=["datetime"], index_col="datetime")
df = df.sort_index()
y = df["close"]

# 2. Exploratory decomposition (STL with weekly seasonality; adjust period if needed)
stl = STL(y, period=7)
res = stl.fit()
trend, seasonal, resid = res.trend, res.seasonal, res.resid

# 3. Compute strength of trend and seasonality
strength_trend = max(0, min(1, 1 - np.var(resid) / np.var(trend + resid)))
strength_seasonal = max(0, min(1, 1 - np.var(resid) / np.var(seasonal + resid)))
print(f"Trend strength:    {strength_trend:.3f}")
print(f"Seasonal strength: {strength_seasonal:.3f}")

# 4. Box–Cox transform
lam = boxcox_normmax(y + 1e-6)  # ensure positivity
y_bc = boxcox(y + 1e-6, lmbda=lam)

# 5. Build exogenous regressors
# 5a. Seasonal dummies (monthly)
dummies = pd.get_dummies(df.index.month, prefix="month", drop_first=True)
dummies.index = df.index

# 5b. Fourier terms for annual seasonality (K=10 harmonics)
def fourier_terms(index, period, K):
    t = np.arange(len(index)) + 1
    data = {}
    for k in range(1, K + 1):
        data[f"sin_{k}"] = np.sin(2 * np.pi * k * t / period)
        data[f"cos_{k}"] = np.cos(2 * np.pi * k * t / period)
    return pd.DataFrame(data, index=index)

fourier_df = fourier_terms(df.index, period=365, K=10)
xreg = pd.concat([dummies, fourier_df], axis=1)

# Ensure no NaNs in exog and align with y_bc
xreg = xreg.loc[y.index].fillna(0)

# 6. Select ARIMA order using AIC
best_aic = np.inf
best_order = (1, 0, 0)  # Fallback default

orders_to_try = [(p, d, q) for p in [0, 1, 2] for d in [0, 1] for q in [0, 1, 2]]

for p, d, q in tqdm(orders_to_try, desc="Finding best ARIMA order"):
    try:
        m = ARIMA(y_bc, order=(p, d, q), exog=xreg)
        r = m.fit()
        if r.aic < best_aic:
            best_aic, best_order = r.aic, (p, d, q)
    except Exception as e:
        continue  # Skip failed fits

print(f"Selected ARIMA order: {best_order} with AIC = {best_aic:.1f}")


# 7. Fit final ARIMA model with exogenous regressors
model = ARIMA(y_bc, order=best_order, exog=xreg.values.astype(float))
res_model = model.fit()
print(res_model.summary())

# 8. Diagnostic plots (like checkresiduals)
res_model.plot_diagnostics(figsize=(12, 8))
plt.tight_layout()
plt.show()

# 9. Forecasting
horizon = 30  # number of days to forecast
future_index = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=horizon, freq="D")

# 9a. Build future exogenous regressors
future_dummies = pd.get_dummies(future_index.month, prefix="month", drop_first=True)
future_dummies.index = future_index           # <-- keep the same index as Fourier terms

future_fourier = fourier_terms(future_index, period=365, K=10)

future_xreg = pd.concat([future_dummies, future_fourier], axis=1)
future_xreg = future_xreg.reindex(columns=xreg.columns, fill_value=0)

# 9b. Generate forecasts on Box–Cox scale
fc_bc = res_model.get_forecast(steps=horizon, exog=future_xreg).predicted_mean

# 9c. Invert Box–Cox
def inv_boxcox(transformed, lmbda):
    if lmbda == 0:
        return np.exp(transformed)
    return np.power(transformed * lmbda + 1, 1 / lmbda)

fc = inv_boxcox(fc_bc, lam)

# 10. Plot actual vs forecast
plt.figure(figsize=(10, 5))
plt.plot(df.index, y, label="Observed")
plt.plot(future_index, fc, label="Forecast", linestyle="--")
plt.title("Crude Oil Close Price Forecast")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.tight_layout()
plt.show()
