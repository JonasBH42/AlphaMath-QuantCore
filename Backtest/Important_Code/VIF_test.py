import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor

data_path = "C:/Users/User/Desktop/Projects/Backtest/Csvs/NQ1!_SCALED_1D.csv"
scaled_df = pd.read_csv(data_path)
dxy_path = "C:/Users/User/Desktop/Projects/Backtest/Csvs/DXY.csv"
dxy_df = pd.read_csv(dxy_path)
us10y_path = "C:/Users/User/Desktop/Projects/Backtest/Csvs/US10Y.csv"
us10y_df = pd.read_csv(us10y_path)

scaled_df["ts_event"] = pd.to_datetime(scaled_df["ts_event"]).dt.strftime(
    "%Y-%m-%d %H:%M:%S"
)
dxy_df["ts_event"] = pd.to_datetime(dxy_df["ts_event"]).dt.strftime("%Y-%m-%d %H:%M:%S")


def compute_ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()


def compute_sma(series: pd.Series, length: int) -> pd.Series:
    return series.rolling(window=length).mean()


def compute_smi(df: pd.DataFrame, length: int) -> pd.Series:
    high_n = df["high"].rolling(window=length).max()
    low_n = df["low"].rolling(window=length).min()
    mid = (high_n + low_n) / 2
    half_range = (high_n - low_n) / 2

    half_range = half_range.replace(0, np.nan)
    smi = 100 * (df["close"] - mid) / half_range
    smi = smi.replace([np.inf, -np.inf], np.nan).fillna(0)
    return smi


ema_length = 14
sma_length = 20
smi_length = 10

scaled_df["SMI"] = compute_smi(scaled_df, smi_length)

scaled_df["returns_pct"] = scaled_df["close"].pct_change()

scaled_df["returns"] = (scaled_df["close"] - scaled_df["open"]) / scaled_df["open"]
scaled_df["rolling_std"] = scaled_df["returns"].rolling(window=2).std(ddof=1)

scaled_df["us10y_close"] = us10y_df["close"]

scaled_df["SMA"] = compute_sma(scaled_df["close"], sma_length)
scaled_df["EMA"] = compute_ema(scaled_df["close"], ema_length)
scaled_df["ma's_pct_diff"] = (
    (scaled_df["EMA"] - scaled_df["SMA"]) / scaled_df["SMA"] * 100
)

scaled_df["volume_ma_pct_diff"] = (
    (scaled_df["volume"] - scaled_df["volume"].rolling(window=20).mean())
    / scaled_df["volume"].rolling(window=20).mean()
    * 100
)

scaled_df["dxy_close"] = dxy_df["close"]

features = ["SMI", "returns_pct", "rolling_std", "dxy_close", "ma's_pct_diff", "volume_ma_pct_diff"]
X = scaled_df[features].dropna()

vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

print(vif_data)
