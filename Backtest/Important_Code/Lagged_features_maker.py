import pandas as pd
import numpy as np

def create_lagged_features(df, lags=1):
    df[f"lag_close_{lags}"] = df["close"].shift(lags)
    df[f"lag_return_{lags}"] = df["close"].shift(lags) - df["open"].shift(lags)
    df[f"lag_open_{lags}"] = df["open"].shift(lags)
    df[f"lag_PVT_{lags}"] = df["PVT"].shift(lags)
    df[f"lag_TCI_{lags}"] = df["TCI"].shift(lags)
    df[f"lag_PAF_{lags}"] = df["PAF"].shift(lags)
    df[f"lag_CSI_{lags}"] = df["CSI"].shift(lags)
    df[f"lag_LPI_{lags}"] = df["LPI"].shift(lags)
    df[f"lag_AFFC_{lags}"] = df["AFFC"].shift(lags)
    df[f"lag_CRI_{lags}"] = df["CRI"].shift(lags)
    df[f"lag_MRI_{lags}"] = df["MRI"].shift(lags)
    df[f"lag_NMI_{lags}"] = df["NMI"].shift(lags)
    df[f"lag_volume_{lags}"] = df["volume"].shift(lags)
    df.dropna(inplace=True)  # Drop rows with NaN after shifting
    return df

df = pd.read_csv(
    "C:/Users/User/Desktop/Projects/Backtest/Csvs/Math_csvs/NQ1!_1D_MATH_PARAMS.csv"
)

df = create_lagged_features(df)
df.to_csv(
    "C:/Users/User/Desktop/Projects/Backtest/Csvs/Math_csvs/NQ1!_1D_MATH_PARAMS_WITH_LAGGED.csv",
    index=False,
)