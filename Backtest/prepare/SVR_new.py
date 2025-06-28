import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
import joblib


def compute_ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()


def compute_sma(series: pd.Series, length: int) -> pd.Series:
    return series.rolling(window=length).mean()


def compute_smi(df: pd.DataFrame, length: int) -> pd.Series:
    high_n = df["high"].rolling(window=length).max()
    low_n = df["low"].rolling(window=length).min()
    mid = (high_n + low_n) / 2
    half_range = (high_n - low_n) / 2

    # Replace zeros in the denominator with NaN to avoid division errors
    half_range = half_range.replace(0, np.nan)
    smi = 100 * (df["close"] - mid) / half_range
    smi = smi.replace([np.inf, -np.inf], np.nan).fillna(0)
    return smi


# ----- Read the historical CSV file -----

# data_path = "NQ1_SCALED_1D_Nadaraya_Watson.csv"

data_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "NQ1_SCALED_1D_Nadaraya_Watson.csv"
)
df = pd.read_csv(data_path, parse_dates=["ts_event"])

# ----- Ask the user for indicator lengths -----
ema_length = int(input("Enter EMA length: "))
sma_length = int(input("Enter SMA length: "))
smi_length = int(input("Enter SMI length: "))


# ----- Compute and add the indicator columns -----
df["EMA"] = compute_ema(df["close"], ema_length)
df["SMA"] = compute_sma(df["close"], sma_length)
df["SMI"] = compute_smi(df, smi_length)
# df["Return"] = df["close"] - df["open"]

# ----- Add lagged data columns -----
# We'll create lagged versions for the columns: 'close', 'Nadaraya_Watson_Regression', 'EMA', 'SMA', 'SMI'
lags = [
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    20,
    22,
    24,
    27,
    31,
    36,
    41,
    46,
    51,
    55,
    63,
    69,
    76,
    82,
    90,
    99,
    108,
    118,
    129,
]
cols_to_lag = [
    "close",
    # "Return",
    "Nadaraya_Watson_Regression",
    "EMA",
    "SMA",
    "SMI",
]

for lag in lags:
    for col in cols_to_lag:
        lag_col_name = f"{col}_lag{lag}"
        df[lag_col_name] = df[col].shift(lag)

# ----- Function to create, train, and evaluate an SVR model -----


def train_and_predict_svr(data: pd.DataFrame) -> pd.DataFrame:
    # Select feature columns: all columns that contain '_lag'
    feature_cols = [col for col in data.columns if "_lag" in col]

    # Drop rows with NaN values (due to shifting) for training.
    train_df = data.dropna(subset=feature_cols + ["close"]).copy()

    X = train_df[feature_cols]
    y = train_df["close"]

    # Create and train the SVR model (using default parameters)
    svr_model = SVR(kernel="linear", C=1.0, epsilon=0.1)
    svr_model.fit(X, y)

    # Evaluate on the training data
    y_pred = svr_model.predict(X)
    mae = mean_absolute_error(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    print(f"Training MAE: {mae:.4f}")
    print(f"Training MSE: {mse:.4f}")

    model_filename = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f"svr_model_ema_{ema_length}_sma_{sma_length}.joblib",
    )
    joblib.dump(svr_model, model_filename)

    # Make predictions for all rows that have complete lagged data.
    data["prediction"] = np.nan
    complete_idx = data.dropna(subset=feature_cols).index
    if not complete_idx.empty:
        X_full = data.loc[complete_idx, feature_cols]
        data.loc[complete_idx, "prediction"] = svr_model.predict(X_full)
    else:
        print("No complete rows to predict.")

    return data


# Train the SVR model and add the 'prediction' column to df.
df = train_and_predict_svr(df)

# ----- Merge predictions into the main CSV -----

main_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "NQ1!_MAIN_1D.csv")
main_df = pd.read_csv(main_path, parse_dates=["ts_event"])

# Merge the 'prediction' from df into main_df on the 'ts_event' column.
merged_df = pd.merge(main_df, df[["ts_event", "prediction"]], on="ts_event", how="left")


def inverse_scale(df_scaled: pd.DataFrame, min_low, max_high, params) -> pd.DataFrame:
    df_original = df_scaled.copy()
    df_original[params] = df_scaled[params] / 100 * (max_high - min_low) + min_low
    return df_original


merged_df = inverse_scale(
    merged_df, merged_df["low"].min(), merged_df["high"].max(), ["prediction"]
)

# output_path = "./NQ1!_MAIN_1D_with_predictions.csv"
# merged_df.to_csv(output_path, index=False)

win = 0
loss = 0
total_predictions = 0
profit = 0
losess = 0

for index, row in merged_df.iterrows():
    if not pd.isnull(row["prediction"]):
        if row["prediction"] > row["open"]:
            if (row["close"] - row["open"]) > 0:
                profit += abs(row["close"] - row["open"]) - 1
                win += 1
                total_predictions += 1
            elif (row["close"] - row["open"]) < 0:
                losess += abs(row["close"] - row["open"]) + 1
                loss += 1
                total_predictions += 1
        elif row["prediction"] < row["open"]:
            if (row["close"] - row["open"]) < 0:
                profit += abs(row["close"] - row["open"]) - 1
                win += 1
                total_predictions += 1
            elif (row["close"] - row["open"]) > 0:
                losess += abs(row["close"] - row["open"]) + 1
                loss += 1
                total_predictions += 1


# Print out the evaluation results.
average_profit = profit / total_predictions
average_loss = losess / total_predictions
risk_reward_ratio = average_profit / average_loss
print(f"Wining Ratio: {win/total_predictions:.3%}")
print(f"Risk Reward Ratio: {risk_reward_ratio:.3f}")
print(f"Total: {profit-losess}")
