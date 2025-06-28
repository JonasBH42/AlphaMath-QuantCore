import joblib
import pandas as pd
import numpy as np
import gc
import pickle
import matplotlib.pyplot as plt


def calculate_tema(data, length=25):
    # Calculate the Triple Exponential Moving Average (TEMA) for a given DataFrame.
    ema = data["close"].ewm(span=length, adjust=False).mean()
    ema2 = ema.ewm(span=length, adjust=False).mean()
    ema3 = ema2.ewm(span=length, adjust=False).mean()
    tema = 3 * (ema - ema2) + ema3
    return tema


def calculate_rsi(data, window=14):

    delta = data["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

tema = 29
sma = 25

file_path = "C:/Users/User/Desktop/Projects/Backtest/Csvs/NQ1_SCALED_1D_Nadaraya_Watson.csv"
model_file = f"C:/Users/User/Desktop/Projects/Backtest/Models/svr_model_tema_{tema}_sma{sma}.joblib"
scaled = pd.read_csv(file_path)
main_file = "C:/Users/User/Desktop/Projects/Backtest/Csvs/NQ1!_MAIN_1D.csv"
main = pd.read_csv(main_file)
# main = main.iloc[3:].reset_index(drop=True)

scaled["lag_open_1"] = scaled["open"].shift(1)
scaled["lag_close_1"] = scaled["close"].shift(1)
scaled["lag_return_1"] = scaled["lag_close_1"] - scaled["lag_open_1"]
scaled["lag_Nadaraya_Watson_Regression"] = scaled["Nadaraya_Watson_Regression"].shift(1)

# scaled["lag_high_1"] = scaled["high"].shift(1)
# scaled["lag_low_1"] = scaled["low"].shift(1)


scaled[f"TEMA_{tema}"] = calculate_tema(scaled, length=tema)
scaled[f"lag_TEMA_{tema}"] = scaled[f"TEMA_{tema}"].shift(1)

scaled[f"sma_{sma}"] = scaled["close"].rolling(window=sma).mean()
scaled[f"lag_sma_{sma}"] = scaled[f"sma_{sma}"].shift(1)

# scaled["RSI"] = calculate_rsi(scaled, window=14)
# scaled["lag_RSI"] = scaled["RSI"].shift(1)

scaled = scaled.drop(
    columns=[
        f"TEMA_{tema}",
        f"sma_{sma}",
        # "RSI",
    ]
)

with open(model_file, "rb") as f:
    model = joblib.load(model_file)

features = [
    "lag_close_1",
    "lag_open_1",
    "lag_return_1",
    "lag_Nadaraya_Watson_Regression",
    f"lag_sma_{sma}",
    f"lag_TEMA_{tema}",
    # "lag_RSI",
]
scaled = scaled.dropna(subset=features)
scaled["prediction"] = model.predict(scaled[features])


# Calculate min_original and max_original from all OHLC columns in main_file

scaled["ts_event"] = pd.to_datetime(scaled["ts_event"])
main["ts_event"] = pd.to_datetime(main["ts_event"])

# Merge predictions into main DataFrame based on 'ts_event'
main = main.merge(scaled[["ts_event", "prediction"]], on="ts_event", how="left")


min_original = main[["open", "high", "low", "close"]].min().min()
max_original = main[["open", "high", "low", "close"]].max().max()
min_scaled = 0
max_scaled = 100
def reverse_scaling(scaled_value):
    scale_factor = (max_original - min_original) / (max_scaled - min_scaled)
    return scaled_value * scale_factor + min_original


# Perform reverse scaling on all 'prediction' values in main
main["prediction"] = main["prediction"].apply(reverse_scaling)


win = 0
loss = 0
total_predictions = 0
profit = 0
losess = 0
avg_loss_drawdown = 0
avg_profit_drawdown = 0
drawdown = 0.0


for index, row in scaled.iterrows():
    if not pd.isnull(row["prediction"]):
        main_row = main.iloc[index]
        if row["prediction"] > row["open"]:
            if (row["close"] - row["open"]) > 0:
                profit += abs(main_row["close"] - main_row["open"]) - 1
                win += 1
                total_predictions += 1
            elif (row["close"] - row["open"]) < 0:
                losess += abs(main_row["close"] - main_row["open"]) + 1
                loss += 1
                total_predictions += 1
        elif row["prediction"] < row["open"]:
            if (row["close"] - row["open"]) < 0:
                profit += abs(main_row["close"] - main_row["open"]) - 1
                win += 1
                total_predictions += 1
            elif (row["close"] - row["open"]) > 0:
                losess += abs(main_row["close"] - main_row["open"]) + 1
                loss += 1
                total_predictions += 1


prediction_csv = "C:/Users/User/Desktop/Projects/Backtest/Csvs/NQ1!_PREDICTION_1D.csv"
scaled.to_csv(prediction_csv, index=False)

if total_predictions > 0:
    win_ratio = win / total_predictions
    avg_loss = losess / total_predictions
    avg_profit = profit / total_predictions
else:
    win_ratio = 0

print(f"Total predictions: {total_predictions}")
print(f"Win count: {win}")
print(f"Win ratio: {win_ratio:.4%}")
print(f"points: {profit-losess}")
print(f"profits: {profit}")
print(f"RR ratio: {avg_profit/avg_loss}")
print(f"avg profit drawdown: {avg_profit_drawdown / win}")
print(f"avg loss drawdown: {avg_loss_drawdown / loss}")


gc.collect()
