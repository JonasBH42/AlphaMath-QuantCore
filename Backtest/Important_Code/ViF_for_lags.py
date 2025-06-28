import pandas as pd
import numpy as np
from arch import arch_model
from statsmodels.stats.outliers_influence import variance_inflation_factor


data_path = "C:/Users/User/Desktop/Projects/Backtest/Csvs/NQ1!_SCALED_1D.csv"
scaled_df = pd.read_csv(data_path)
es_path = "C:/Users/User/Desktop/Projects/Backtest/Csvs/ES1!_1D_scaled.csv"
es_df = pd.read_csv(es_path)
dxy_path = "C:/Users/User/Desktop/Projects/Backtest/Csvs/DXY.csv"
dxy_df = pd.read_csv(dxy_path)
alt_df = pd.read_csv("C:/Users/User/Desktop/Projects/Backtest/Csvs/ZN_1D.csv")
vix_df = pd.read_csv("C:/Users/User/Desktop/Projects/Backtest/Csvs/VIX.csv")
us10y_df = pd.read_csv("C:/Users/User/Desktop/Projects/Backtest/Csvs/US10Y.csv")
us2y_df = pd.read_csv("C:/Users/User/Desktop/Projects/Backtest/Csvs/US2Y.csv")
rty_df = pd.read_csv("C:/Users/User/Desktop/Projects/Backtest/Csvs/IWM_scaled.csv")
xlf_df = pd.read_csv("C:/Users/User/Desktop/Projects/Backtest/Csvs/XLF_scaled.csv")
xlk_df = pd.read_csv("C:/Users/User/Desktop/Projects/Backtest/Csvs/XLK_scaled.csv")
nadq_df = pd.read_csv("C:/Users/User/Desktop/Projects/Backtest/Csvs/NDAQ.csv")
options_df = pd.read_csv(
    "C:/Users/User/Desktop/Projects/Backtest/Csvs/qqq_options_scaled.csv"
)


scaled_df["ts_event"] = pd.to_datetime(scaled_df["ts_event"]).dt.strftime(
    "%Y-%m-%d %H:%M:%S"
)
dxy_df["ts_event"] = pd.to_datetime(dxy_df["ts_event"]).dt.strftime("%Y-%m-%d %H:%M:%S")
alt_df["ts_event"] = pd.to_datetime(alt_df["ts_event"]).dt.strftime("%Y-%m-%d %H:%M:%S")
vix_df["ts_event"] = pd.to_datetime(vix_df["ts_event"]).dt.strftime("%Y-%m-%d %H:%M:%S")
us10y_df["ts_event"] = pd.to_datetime(us10y_df["ts_event"]).dt.strftime(
    "%Y-%m-%d %H:%M:%S"
)
us2y_df["ts_event"] = pd.to_datetime(us2y_df["ts_event"]).dt.strftime(
    "%Y-%m-%d %H:%M:%S"
)
xlf_df["ts_event"] = pd.to_datetime(xlf_df["ts_event"]).dt.strftime("%Y-%m-%d %H:%M:%S")
xlk_df["ts_event"] = pd.to_datetime(xlk_df["ts_event"]).dt.strftime("%Y-%m-%d %H:%M:%S")
nadq_df["ts_event"] = pd.to_datetime(nadq_df["ts_event"]).dt.strftime(
    "%Y-%m-%d %H:%M:%S"
)
options_df["ts_event"] = pd.to_datetime(options_df["ts_event"]).dt.strftime(
    "%Y-%m-%d %H:%M:%S"
)
rty_df["ts_event"] = pd.to_datetime(rty_df["ts_event"]).dt.strftime("%Y-%m-%d %H:%M:%S")

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


# scaled_df["returns_pct"] = scaled_df["close"].pct_change()

# scaled_df["returns"] = (scaled_df["close"] - scaled_df["open"]) / scaled_df["open"]
# scaled_df["rolling_std"] = scaled_df["returns"].rolling(window=2).std(ddof=1)


def merge_series_to_df(df, series_df, series_name):
    df[f"{series_name}"] = np.nan
    orig_df_z_scores = dict(zip(series_df["ts_event"], series_df[f"{series_name}"]))
    df[f"{series_name}"] = df["ts_event"].map(orig_df_z_scores)
    df[f"{series_name}"] = df[f"{series_name}"].ffill()
    return df


def compute_rolling_std_zscore(df, orig_df, price_col, window, ddof=1):

    df["rolling_mean"] = df["close"].rolling(window=window, min_periods=1).mean()
    df["rolling_std"] = df["close"].rolling(window=window, min_periods=1).std(ddof=ddof)

    df["z_score"] = (df["close"] - df["rolling_mean"]) / df["rolling_std"]
    orig_df[f"{price_col}"] = np.nan
    orig_df_z_scores = dict(zip(df["ts_event"], df["z_score"]))
    orig_df[f"{price_col}"] = orig_df["ts_event"].map(orig_df_z_scores)
    orig_df[f"{price_col}"] = orig_df[f"{price_col}"].ffill()
    return orig_df


scaled_df["dxy_realised"] = dxy_df["close"].rolling(window=20).std(ddof=5)
scaled_df["es_close"] = es_df["close"]


dxy_df["realised_vol"] = dxy_df["close"].rolling(window=5).std(ddof=1)
dxy_df["implied_vol"] = dxy_df["close"] / dxy_df["realised_vol"]
scaled_df["dxy_implied_vol"] = dxy_df["implied_vol"]


threshold = 20
vix_df["vix_deviation_threshold"] = (vix_df["close"] - threshold).abs()
scaled_df = merge_series_to_df(scaled_df, vix_df, "vix_deviation_threshold")


scaled_df["implied_vol"] = scaled_df["close"].rolling(window=5).std(ddof=1)
vix_df["vix_close"] = vix_df["close"]
scaled_df = merge_series_to_df(scaled_df, vix_df, "vix_close")
scaled_df["vix_div_implied_vol"] = vix_df["vix_close"] / scaled_df["implied_vol"]


scaled_df["bonds_diff"] = us10y_df["close"] - us2y_df["close"]


scaled_df["es_realised"] = scaled_df["es_close"].rolling(window=21).std(ddof=1)
scaled_df["es_implied"] = scaled_df["es_close"] / scaled_df["es_realised"]
scaled_df["es_risk_premium"] = scaled_df["es_implied"] / scaled_df["es_realised"]


options_df["options_volume"] = options_df["volume"].rolling(window=5).std(ddof=1)
scaled_df = merge_series_to_df(scaled_df, options_df, "options_volume")


nadq_df["nq_close"] = nadq_df["close"]
scaled_df = merge_series_to_df(scaled_df, nadq_df, "nq_close")
scaled_df["nadq_diff"] = scaled_df["close"] - scaled_df["nq_close"]






rty_df["rty_close"] = rty_df["close"]
scaled_df = merge_series_to_df(scaled_df, rty_df, "rty_close")
scaled_df["rty_div_close"] = (scaled_df["rty_close"] / scaled_df["close"]).rolling(5).std(ddof=1)

xlk_df["xlk_realised_vol"] = xlk_df["close"].rolling(window=20).std(ddof=1)
xlk_df["implied_vol"] = xlk_df["close"] / xlk_df["xlk_realised_vol"]
scaled_df = merge_series_to_df(scaled_df, xlk_df, "implied_vol")
scaled_df["xlk_implied_div"] = scaled_df["implied_vol"] / scaled_df["close"]

lagged_col_names = {
    "vix_div_implied_vol": [1, 15],
    "vix_deviation_threshold": [1],
    "es_close": [1],
    "bonds_diff": [1],
    "nadq_diff": [1],
    "dxy_realised": [1],
    "rty_div_close": [1],
    "xlk_implied_div": [1],
    "options_volume": [1],

    # "xlk_div_nadq": [1],
    # "es_risk_premium": [1],
    # "vix_z_score": [1, 6, 23, 29],
}
for col, lags in lagged_col_names.items():
    if isinstance(lags, list):
        for lag in lags:
            scaled_df[f"{col}_lag_{lag}"] = scaled_df[col].shift(lag)
    else:
        scaled_df[f"{col}_lag_{lags}"] = scaled_df[col].shift(lags)


features = [col for col in scaled_df.columns if "lag" in col]
X = scaled_df[features].dropna()


vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
vif_data["VIF"] = [
    variance_inflation_factor(X.values, i) for i in range(len(X.columns))
]

print(vif_data)
