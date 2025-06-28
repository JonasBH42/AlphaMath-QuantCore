import pandas as pd
import numpy as np


def convert_columns_to_numeric(df_path, columns_to_convert, strings_to_remove):
    df = pd.read_csv(df_path)
    df_copy = df.copy()

    for column in columns_to_convert:
        if column in df_copy.columns:
            if df_copy[column].dtype == object:
                for string_to_remove in strings_to_remove:
                    df_copy[column] = df_copy[column].str.replace(
                        string_to_remove, "", regex=False
                    )

                df_copy[column] = pd.to_numeric(df_copy[column], errors="coerce")
        else:
            print(f"Warning: Column '{column}' not found in DataFrame")
    if "ts_event" in columns_to_convert:
        df_copy["ts_event"] = pd.to_datetime(df["ts_event"]).dt.strftime(
            "%Y-%m-%dT%H:%M:%S"
        )

    df_copy.to_csv(df_path, index=False)


path = "C:/Users/User/Desktop/Projects/Backtest/Csvs/US10Y.csv"
result_df = convert_columns_to_numeric(
    path, ["close", "open", "high", "low", "change", "ts_event"], ['"', "'", "%"]
)
