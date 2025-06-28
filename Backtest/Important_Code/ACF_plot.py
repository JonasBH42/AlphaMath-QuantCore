import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.graphics.tsaplots import plot_acf
import matplotlib.pyplot as plt
import statsmodels.api as sm
from arch import arch_model


data_path = "C:/Users/User/Desktop/Projects/Backtest/Csvs/NQ1!_SCALED_1D.csv"
scaled_df = pd.read_csv(data_path)
es_path = "C:/Users/User/Desktop/Projects/Backtest/Csvs/ES1!_1D_scaled.csv"
es_df = pd.read_csv(es_path)
dxy_path = "C:/Users/User/Desktop/Projects/Backtest/Csvs/DXY_SCALED_1D.csv"
dxy_df = pd.read_csv(dxy_path)
zn_path = "C:/Users/User/Desktop/Projects/Backtest/Csvs/ZN_1D_scaled.csv"
zn_path = pd.read_csv(zn_path)
vix_df = pd.read_csv("C:/Users/User/Desktop/Projects/Backtest/Csvs/VIX_scaled.csv")
us10y_df = pd.read_csv("C:/Users/User/Desktop/Projects/Backtest/Csvs/US10Y.csv")
us2y_df = pd.read_csv("C:/Users/User/Desktop/Projects/Backtest/Csvs/US2Y.csv")

xlf_df = pd.read_csv("C:/Users/User/Desktop/Projects/Backtest/Csvs/XLF_scaled.csv")
xlk_df = pd.read_csv("C:/Users/User/Desktop/Projects/Backtest/Csvs/XLK_scaled.csv")
rty_df = pd.read_csv("C:/Users/User/Desktop/Projects/Backtest/Csvs/IWM_scaled.csv")
nadq_df = pd.read_csv("C:/Users/User/Desktop/Projects/Backtest/Csvs/NDAQ.csv")

scaled_df["ts_event"] = pd.to_datetime(scaled_df["ts_event"]).dt.strftime(
    "%Y-%m-%d %H:%M:%S"
)

dxy_df["ts_event"] = pd.to_datetime(dxy_df["ts_event"]).dt.strftime("%Y-%m-%d %H:%M:%S")
zn_path["ts_event"] = pd.to_datetime(zn_path["ts_event"]).dt.strftime("%Y-%m-%d %H:%M:%S")
vix_df["ts_event"] = pd.to_datetime(vix_df["ts_event"]).dt.strftime("%Y-%m-%d %H:%M:%S")
us10y_df["ts_event"] = pd.to_datetime(us10y_df["ts_event"]).dt.strftime(
    "%Y-%m-%d %H:%M:%S"
)
us2y_df["ts_event"] = pd.to_datetime(us2y_df["ts_event"]).dt.strftime(
    "%Y-%m-%d %H:%M:%S"
)
xlf_df["ts_event"] = pd.to_datetime(xlf_df["ts_event"]).dt.strftime("%Y-%m-%d %H:%M:%S")
xlk_df["ts_event"] = pd.to_datetime(xlk_df["ts_event"]).dt.strftime("%Y-%m-%d %H:%M:%S")
rty_df["ts_event"] = pd.to_datetime(rty_df["ts_event"]).dt.strftime("%Y-%m-%d %H:%M:%S")
nadq_df["ts_event"] = pd.to_datetime(nadq_df["ts_event"]).dt.strftime(
    "%Y-%m-%d %H:%M:%S"
)


ema_length = 14
sma_length = 20
smi_length = 10



# def dynamic_bol_vol(scaled_df, price_col='close', hurst_window=60):
    
#     df = scaled_df.copy()  # So we don't mutate the original data
#     df['bol_vol'] = np.nan  # Prepare the output column
    
#     prices = df[price_col]
#     n = len(df)
    
#     for i in range(hurst_window, n):
#         # 1) Compute Hurst exponent over the past 'hurst_window' days
#         window_data = prices.iloc[i - hurst_window : i].dropna().values
        
#         # If there's insufficient data, skip
#         if len(window_data) < 20:
#             continue
        
#         # -- Hurst exponent calculation (simplified R/S on lags 2..19) --
#         lags = range(2, 20)
#         tau = []
#         for lag in lags:
#             diff = window_data[lag:] - window_data[:-lag]
#             tau.append(np.sqrt(np.std(diff)))
        
#         # Convert to log-log and fit
#         lags_log = np.log(np.array(list(lags)))
#         tau_log = np.log(np.array(tau))
        
#         if np.any(np.isinf(tau_log)) or np.any(np.isnan(tau_log)):
#             # If we can't compute a valid H, skip
#             continue
        
#         reg = np.polyfit(lags_log, tau_log, 1)
#         H_val = reg[0]
        
#         # 2) Choose Bollinger window
#         if H_val > 0.7:
#             w = 10
#         elif H_val < 0.3:
#             w = 30
#         else:
#             w = 20
        
#         # 3) Compute Bollinger Bands for the chosen window
#         if i - w < 0:
#             continue
        
#         w_data = prices.iloc[i - w : i]
#         ma = w_data.mean()
#         std = w_data.std()
        
#         upper = ma + 2.0 * std
#         lower = ma - 2.0 * std
        
#         # 4) Store the band-width in 'bol_vol'
#         df.loc[df.index[i], 'bol_vol'] = upper - lower

#     return df

# scaled_df = dynamic_bol_vol(scaled_df)


def merge_series_to_df(df, series_df, series_name):
    df[f"{series_name}"] = np.nan
    orig_df_z_scores = dict(zip(series_df["ts_event"], series_df[f"{series_name}"]))
    df[f"{series_name}"] = df["ts_event"].map(orig_df_z_scores)
    df[f"{series_name}"] = df[f"{series_name}"].ffill()
    return df



scaled_df["returns_diff"] = scaled_df["close"].pct_change()


window = 80
zn_path["rolling_mean"] = zn_path["close"].rolling(window=window, min_periods=1).mean()
zn_path["rolling_std"] = zn_path["close"].rolling(window=window, min_periods=1).std(ddof=30)
zn_path["z_score"] = (zn_path["close"] - zn_path["rolling_mean"]) / zn_path["rolling_std"]

zn_path["diff"] = zn_path["close"] - zn_path["open"]

scaled_df["alt"] = np.nan 
alt_z_scores = dict(zip(zn_path["ts_event"], zn_path["z_score"]))
scaled_df["alt"] = scaled_df["ts_event"].map(alt_z_scores)
scaled_df["alt"] = scaled_df["alt"].ffill()

scaled_df["dxy"] = dxy_df["close"].rolling(window=20).std(ddof=5)


scaled_df["implied_vol"] = scaled_df["close"].rolling(window=5).std(ddof=1)
vix_df["vix_close"] = vix_df["close"]
scaled_df = merge_series_to_df(scaled_df, vix_df, "vix_close")
scaled_df["vix_div_implied_vol"] = vix_df["vix_close"] / scaled_df["implied_vol"]

threshold = 40
scaled_df["vix_close"] = (vix_df["close"] - threshold).abs().rolling(window=20).std(ddof=1)

lagged_cols = ["vix_div_implied_vol"]

for col in lagged_cols:
    plot_acf(scaled_df[col].dropna(), lags=100)
    plt.title(f"ACF for {col}")
    plt.show()
