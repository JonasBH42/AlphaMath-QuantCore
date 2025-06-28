import pandas as pd
import numpy as np


def scale_and_save_df(file_name, output_name):

    df = pd.read_csv(f"C:/Users/User/Desktop/Projects/Backtest/Csvs/{file_name}.csv")
    df_scaled = df.copy()

    columns_to_drop = [
        "volume",
        "symbol",
        "ts_event",
        "rtype",
        "publisher_id",
        "instrument_id",
    ]
    df_scaled = df_scaled.drop(
        columns=[col for col in columns_to_drop if col in df.columns]
    )

    df_scaled = df_scaled.apply(pd.to_numeric, errors="coerce")

    low_min = df_scaled["low"].min()
    high_max = df_scaled["high"].max()

    df_scaled = (df_scaled - low_min) / (high_max - low_min) * 100

    # scaled_df["volume"] = df["volume"]
    df_scaled.insert(0, "ts_event", df["ts_event"])

    output_path = f"C:/Users/User/Desktop/Projects/Backtest/Csvs/{output_name}.csv"
    df_scaled.to_csv(output_path, index=False)

    return df_scaled


file_path = "cpi"
output_name = "CPI_scaled"
df = pd.read_csv(f"C:/Users/User/Desktop/Projects/Backtest/Csvs/{file_path}.csv")
df_scale = df.copy().drop(columns=["Year", "Avg","Dec-Dec","Avg-Avg"])
df_scale = df_scale.apply(pd.to_numeric, errors="coerce")
min = df_scale.min().min()
max = df_scale.max().max()
df_scale = (df_scale - min) / (max - min) * 100
df_scale.insert(0, "Year", df["Year"])
df_scale.to_csv(
    f"C:/Users/User/Desktop/Projects/Backtest/Csvs/{file_path}_scaled.csv", index=False
)
# scale_and_save_df(file_path, output_name)
