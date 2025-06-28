import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from scipy.stats import norm
import joblib
from sklearn.model_selection import GridSearchCV


data_path = "C:/Users/User/Desktop/Projects/Backtest/Csvs/NQ1!_SCALED_1D.csv"
scaled_df = pd.read_csv(data_path)
es_path = "C:/Users/User/Desktop/Projects/Backtest/Csvs/ES1!_1D_scaled.csv"
es_df = pd.read_csv(es_path)
main_path = (
    "C:/Users/User/Desktop/Projects/Backtest/Csvs/NQ1!_1D_with_volatility_gjr.csv"
)
main_df = pd.read_csv(main_path)
dxy_path = "C:/Users/User/Desktop/Projects/Backtest/Csvs/DXY_SCALED_1D.csv"
dxy_df = pd.read_csv(dxy_path)
alt_path = "C:/Users/User/Desktop/Projects/Backtest/Csvs/ZN_1D.csv"
alt_df = pd.read_csv(alt_path)
vix_path = "C:/Users/User/Desktop/Projects/Backtest/Csvs/VIX.csv"
vix_df = pd.read_csv(vix_path)
us10y_df = pd.read_csv("C:/Users/User/Desktop/Projects/Backtest/Csvs/US10Y.csv")
us2y_df = pd.read_csv("C:/Users/User/Desktop/Projects/Backtest/Csvs/US2Y.csv")
xlf_df = pd.read_csv("C:/Users/User/Desktop/Projects/Backtest/Csvs/XLF_scaled.csv")
xlk_df = pd.read_csv("C:/Users/User/Desktop/Projects/Backtest/Csvs/XLK_scaled.csv")
rty_df = pd.read_csv("C:/Users/User/Desktop/Projects/Backtest/Csvs/IWM_scaled.csv")
nadq_df = pd.read_csv("C:/Users/User/Desktop/Projects/Backtest/Csvs/NDAQ.csv")
options_df = pd.read_csv(
    "C:/Users/User/Desktop/Projects/Backtest/Csvs/qqq_options_scaled.csv"
)
cpi_df = pd.read_csv("C:/Users/User/Desktop/Projects/Backtest/Csvs/cpi_scaled.csv")

scaled_df["ts_event"] = pd.to_datetime(scaled_df["ts_event"]).dt.strftime(
    "%Y-%m-%d %H:%M:%S"
)
main_df["ts_event"] = pd.to_datetime(main_df["ts_event"]).dt.strftime(
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
rty_df["ts_event"] = pd.to_datetime(rty_df["ts_event"]).dt.strftime("%Y-%m-%d %H:%M:%S")
nadq_df["ts_event"] = pd.to_datetime(nadq_df["ts_event"]).dt.strftime(
    "%Y-%m-%d %H:%M:%S"
)
options_df["ts_event"] = pd.to_datetime(options_df["ts_event"]).dt.strftime(
    "%Y-%m-%d %H:%M:%S"
)


# def dynamic_bol_vol(scaled_df, price_col="close", hurst_window=60):

#     df = scaled_df.copy()  # So we don't mutate the original data
#     df["bol_vol"] = np.nan  # Prepare the output column

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
#         df.loc[df.index[i], "bol_vol"] = upper - lower

#     return df

# scaled_df = dynamic_bol_vol(scaled_df)


def merge_series_to_df(df, series_df, series_name):
    df[f"{series_name}"] = np.nan
    orig_df_z_scores = dict(zip(series_df["ts_event"], series_df[f"{series_name}"]))
    df[f"{series_name}"] = df["ts_event"].map(orig_df_z_scores)
    df[f"{series_name}"] = df[f"{series_name}"].ffill()
    return df


def compute_rolling_std_zscore(df, orig_df, price_col, window, ddof=1):

    df["rolling_mean"] = df["close"].rolling(window=window, min_periods=1).mean()
    df["rolling_std"] = df["close"].rolling(window=window, min_periods=1).std(ddof=ddof)

    df[f"{price_col}"] = (df["close"] - df["rolling_mean"]) / df["rolling_std"]
    orig_df = merge_series_to_df(orig_df, df, price_col)
    return orig_df


scaled_df["dxy_realised"] = dxy_df["close"].rolling(window=20).std(ddof=5)

scaled_df["es_close"] = es_df["close"]


nadq_df["nq_close"] = nadq_df["close"]
scaled_df = merge_series_to_df(scaled_df, nadq_df, "nq_close")
scaled_df["nadq_diff"] = scaled_df["close"] - scaled_df["nq_close"]
decay_factors = np.exp(-np.arange(1, 11))
scaled_df["nadq_diff_decay"] = (
    scaled_df["nadq_diff"]
    .rolling(window=5)
    .apply(lambda x: np.sum(x * decay_factors[-len(x) :]), raw=True)
)


dxy_df["realised_vol"] = dxy_df["close"].rolling(window=5).std(ddof=1)
dxy_df["implied_vol"] = dxy_df["close"] / dxy_df["realised_vol"]
scaled_df["dxy_implied_vol"] = dxy_df["implied_vol"]


us10y_df["bonds_diff"] = us10y_df["close"] / us2y_df["close"]
scaled_df = merge_series_to_df(scaled_df, us10y_df, "bonds_diff")

rty_df["rty_close"] = rty_df["close"]
scaled_df = merge_series_to_df(scaled_df, rty_df, "rty_close")
scaled_df["rty_div_close"] = (
    (scaled_df["rty_close"] / scaled_df["close"]).rolling(20).std(ddof=1)
)

threshold = 20
vix_df["vix_deviation_threshold"] = (vix_df["close"] - threshold).abs()
scaled_df = merge_series_to_df(scaled_df, vix_df, "vix_deviation_threshold")


scaled_df["realised_vol"] = scaled_df["close"].rolling(window=5).std(ddof=1)
vix_df["vix_close"] = vix_df["close"]
scaled_df = merge_series_to_df(scaled_df, vix_df, "vix_close")
scaled_df["vix_implied_vol"] = scaled_df["vix_close"] / scaled_df["realised_vol"]


scaled_df["es_realised"] = scaled_df["es_close"].rolling(window=15).std(ddof=5)
scaled_df["es_implied"] = scaled_df["es_close"] / scaled_df["es_realised"]
scaled_df["es_risk_premium"] = scaled_df["es_implied"] / scaled_df["es_realised"]


scaled_df = compute_rolling_std_zscore(vix_df, scaled_df, "vix_z_score", 40, 30)


cpi_df = cpi_df.set_index("Year")
cpi_df.columns = [
    pd.to_datetime(month, format="%b").month if not month.isdigit() else int(month)
    for month in cpi_df.columns
]


def map_cpi_to_scaled_df(row, cpi_df):
    year = pd.to_datetime(row["ts_event"]).year
    month = pd.to_datetime(row["ts_event"]).month
    try:
        if month == 1:  # If it's January, use December of the previous year
            return cpi_df.loc[year - 1, 12]
        else:  # Otherwise, use the previous month of the same year
            return cpi_df.loc[year, month - 1]
    except KeyError:
        return np.nan


scaled_df["cpi"] = scaled_df.apply(map_cpi_to_scaled_df, axis=1, cpi_df=cpi_df)


options_df["options_volume"] = options_df["volume"].rolling(window=20).std(ddof=1)
scaled_df = merge_series_to_df(scaled_df, options_df, "options_volume")


xlk_df["xlk_realised_vol"] = xlk_df["close"].rolling(window=20).std(ddof=1)
xlk_df["implied_vol"] = xlk_df["close"] / xlk_df["xlk_realised_vol"]
scaled_df = merge_series_to_df(scaled_df, xlk_df, "implied_vol")
scaled_df["xlk_implied_div"] = scaled_df["implied_vol"] / scaled_df["close"]


def calculate_trix(df, column="close", period=15):
    # Calculate the TRIX indicator
    df["ema1"] = df[column].ewm(span=period, adjust=False).mean(5)
    df["ema2"] = df["ema1"].ewm(span=period, adjust=False).mean(5)
    df["ema3"] = df["ema2"].ewm(span=period, adjust=False).mean(5)
    df["trix"] = df["ema3"].pct_change() * 100
    return df


scaled_df = calculate_trix(scaled_df, column="close", period=18)
scaled_df = scaled_df.drop(
    columns=["ema1", "ema2", "ema3"]
)  # Drop intermediate columns


lagged_col_names = {
    "vix_implied_vol": [1, 15],
    "vix_deviation_threshold": [1],
    "es_close": [1],
    "bonds_diff": [1],
    "nadq_diff": [1],
    "dxy_realised": [1],
    "rty_div_close": [1],
    "xlk_implied_div": [1],
    "options_volume": [1],
    "trix": [1],
}

for col, lags in lagged_col_names.items():
    if isinstance(lags, list):
        for lag in lags:
            scaled_df[f"{col}_lag_{lag}"] = scaled_df[col].shift(lag)
    else:
        scaled_df[f"{col}_lag_{lags}"] = scaled_df[col].shift(lags)


scaled_df = scaled_df.dropna()
# scaled_df = scaled_df.drop(columns=["ts_event"])

y = scaled_df["close"].values

X = scaled_df[
    [col for col in scaled_df.columns if ("lag" in col or col == "ts_event")]
].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=False
)

ts_event_test = scaled_df["ts_event"].iloc[X_train.shape[0] :]
X_train = np.delete(X_train, np.where(scaled_df.columns == "ts_event"), axis=1)
X_test = np.delete(X_test, np.where(scaled_df.columns == "ts_event"), axis=1)

# Define the parameter grid for Ridge regression
param_grid = {"alpha": np.logspace(-4, 4, 50)}

# Perform cross-validation using GridSearchCV
ridge_cv = GridSearchCV(Ridge(), param_grid, scoring="neg_mean_squared_error", cv=5)
ridge_cv.fit(X_train, y_train)

# Update the model with the best alpha
best_alpha = ridge_cv.best_params_["alpha"]
model = Ridge(alpha=best_alpha)
model.fit(X_train, y_train)

# window_size = 100
# for start in range(0, len(X_train) - window_size + 1):
#     end = start + window_size
#     if end > len(X_train):
#         break
#     X_train_window = X_train[start:end]
#     y_train_window = y_train[start:end]
#     model.fit(X_train_window, y_train_window)


predictions_test = model.predict(X_test)


mse = mean_squared_error(y_test, predictions_test)
mae = mean_absolute_error(y_test, predictions_test)
r2 = r2_score(y_test, predictions_test)


print("Coefficients per feature:")
feature_names = [col for col in scaled_df.columns if "lag" in col]
coefficients = model.coef_
for feature, coef in zip(feature_names, coefficients):
    print(f"{feature}: {coef}")

print("Mean Squared Error (MSE):", mse)
print("Mean Absolute Error (MAE):", mae)
print("R^2 Score:", r2)


plt.figure(figsize=(14, 7))
# plt.plot(scaled_nq["close"], label="Close Prices", color="green")
plt.plot(y_test, label="Actual Close Prices", color="blue")
plt.plot(predictions_test, label="Predicted Close Prices", color="red")
plt.title("Actual vs Predicted Close Prices")
plt.xlabel("Time")
plt.ylabel("Close Price")
plt.legend()
plt.show()


merged_df = main_df[main_df["ts_event"].isin(ts_event_test)].copy()


low_min = main_df["low"].min()
high_max = main_df["high"].max()

# Descale the predictions back to the original size
merged_df["prediction"] = predictions_test
merged_df["prediction"] = merged_df["prediction"] * (high_max - low_min) / 100 + low_min

merged_df.index = ts_event_test.index

win = 0
losess = 0
total_predictions = 0
profit = 0
loss = 0

for index, row in merged_df.iterrows():
    if not pd.isnull(row["prediction"]):
        if row["prediction"] > row["open"]:
            if (row["close"] - row["open"]) < 0 or abs(row["low"] - row["open"]) >= row["volatility"] * 1:
                loss += min(abs(row["close"] - row["open"]), row["volatility"] * 1)
                losess += 1
                total_predictions += 1
            elif (row["close"] - row["open"]) > 0:
                profit += abs(row["close"] - row["open"])
                win += 1
                total_predictions += 1
        elif row["prediction"] < row["open"]:
            if (row["close"] - row["open"]) > 0 or abs(row["open"] - row["high"]) >= row["volatility"] * 1:
                loss += min(abs(row["close"] - row["open"]), row["volatility"] * 1)
                losess += 1
                total_predictions += 1
            elif (row["close"] - row["open"]) < 0:
                profit += abs(row["close"] - row["open"])
                win += 1
                total_predictions += 1
            


average_profit = profit / total_predictions
average_loss = loss / total_predictions
risk_reward_ratio = average_profit / average_loss
print("Evaluation of Predictions:")
print(f"Total Predictions: {total_predictions}")
print(f"Total wins: {win}")
print(f"Total profit: {profit}")
print(f"Average profit: {average_profit}")
print(f"Total loss: {loss}")
print(f"Total losess: {losess}")
print(f"Wining Ratio: {win/total_predictions:.3%}")
print(f"Risk Reward Ratio: {risk_reward_ratio:.3f}")
print(f"Total: {profit-loss}")
