import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import joblib

# Specify the path to your CSV file
csv_file = "C:/Users/User/Desktop/AlphaMath-QuantCore/Backtest/Csvs/NQ1!_MAIN_1D.csv"

# Read the CSV file; here we read the raw CSV without date parsing initially.
try:
    data = pd.read_csv(csv_file)
except Exception as e:
    print("Error reading CSV file:", e)
    sys.exit()

data["ts_event"] = pd.to_datetime(data["ts_event"]).dt.strftime("%Y-%d-%m %H:%M:%S")
data.rename(columns={"ts_event": "Date"}, inplace=True)
# Print the head for confirmation
print("Retrieved data head:")
print(data["Date"].head(10))

# Convert the "Date" column to datetime with dayfirst=True (since your dates are in day-month-year format)
# and then set it as the index.
try:
    data["Date"] = pd.to_datetime(data["Date"], dayfirst=True)
    data.set_index("Date", inplace=True)
except Exception as e:
    print("Error converting index to datetime:", e)
    sys.exit()


# Filter the data from 2015 to 2024.
data = data.loc[(data.index.year >= 2014) & (data.index.year <= 2025)]
print("\nFiltered data range:", data.index.min(), "to", data.index.max())

# -----------------------
# Feature Engineering
# -----------------------

# Convert column names to lower case for consistency
data.columns = [col.lower() for col in data.columns]

# Assume the CSV already contains a "return" column in points.
# Do NOT recalculate return using pct_change(), otherwise you would lose the original points.
# Calculate various Exponential Moving Averages (EMAs) on the close price
data["ema10"] = data["close"].ewm(span=10, adjust=False).mean()
data["ema20"] = data["close"].ewm(span=20, adjust=False).mean()
data["ema50"] = data["close"].ewm(span=50, adjust=False).mean()

# Use the file's 'return' column (which is in points) as is.
# Create lagged return features (1-day and 2-day lags)
data["return"] = (
    data["close"] - data["open"]
)  # Assuming 'return' is the difference in close price
data["lag1_return"] = data["return"].shift(1)
data["lag2_return"] = data["return"].shift(2)

# Rolling metrics: rolling standard deviation of the return (in points) over 10-day and 20-day windows
data["roll_std_10"] = data["return"].rolling(window=10).std()
data["roll_std_20"] = data["return"].rolling(window=20).std()

# Additional feature: lagged close price
data["lag1_close"] = data["close"].shift(1)

# Drop rows with missing values resulting from shifting and rolling calculations
data = data.dropna()

# -----------------------
# Train-Test Split
# -----------------------
# Define training period: 2015-2020 and testing period: 2021-2024
train = data.loc["2014":"2021"]
test = data.loc["2022":]

print("\nData head after datetime conversion:")
print(test.head())

# Define feature columns - adjust or add features as needed
features = [
    "ema10",
    "ema20",
    "ema50",
    "lag1_return",
    "lag2_return",
    "roll_std_10",
    "roll_std_20",
    "lag1_close",
]

# Define target variable (predicting close)
target = "close"

# Prepare train and test sets for features (X) and target (y)
X_train = train[features]
y_train = train[target]
X_test = test[features]
y_test = test[target]

# -----------------------
# Feature Scaling
# -----------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------
# Lasso Regression Model Training
# -----------------------
lasso = Lasso(alpha=0.01, max_iter=10000)
lasso.fit(X_train_scaled, y_train)

# Import joblib for model saving

# Save the trained Lasso model
model_path = "C:/Users/User/Desktop/AlphaMath-QuantCore/Backtest/lasso_model.joblib"
joblib.dump(lasso, model_path)
print(f"Model saved to: {model_path}")

# Predict on the test set
y_pred = lasso.predict(X_test_scaled)

# -----------------------
# Trading Signal and PnL Calculation for the Test Set
# -----------------------

# Create a copy of the test data to add predictions and trade signals.
results = test.copy()
results["high"] = test["high"]
results["low"] = test["low"]
results["open"] = test["open"]
results["predicted_close"] = y_pred

# Determine trade signal: BUY if predicted close > actual open, SELL otherwise.
results["trade"] = np.where(results["predicted_close"] > results["open"], "BUY", "SELL")

# Calculate PnL: if BUY, use the day's 'return' (points); if SELL, use the negative of the day's 'return'

results['pnl'] = results.apply(lambda row: (-100 if abs(row["low"] - row["open"]) >= 100 else row['return']) if row['trade'] == 'BUY'
                               else (-100 if abs(row["high"] - row["open"]) >= 100 else -row['return']), axis=1)
# results["pnl"] = results.apply(
#     lambda row: row["return"] if row["trade"] == "BUY" else -row["return"], axis=1
# )

# -----------------------
# Additional Metrics: Average Winner, Average Loser, and Risk/Reward Ratio (RRR)
# -----------------------

avg_winner = results.loc[results["pnl"] > 0, "pnl"].mean()
avg_loser = results.loc[results["pnl"] < 0, "pnl"].mean()  # avg_loser will be negative
rrr = avg_winner / abs(avg_loser) if avg_loser != 0 else np.nan

# Print the PnL summary
total_pnl = results["pnl"].sum()
positive_trades = (results["pnl"] > 0).sum()
negative_trades = (results["pnl"] < 0).sum()

print("\nPnL Summary:")
print("Total PnL:", total_pnl)
print("Number of positive PnL values:", positive_trades)
print("Number of negative PnL values:", negative_trades)
print("Average Winner (PnL):", avg_winner)
print("Average Loser (PnL):", avg_loser)
print("Risk Reward Ratio (RRR):", rrr)

# -----------------------
# Model Evaluation
# -----------------------
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\nMean Absolute Error (MAE):", mae)
print("Root Mean Squared Error (RMSE):", rmse)

# -----------------------
# Plot the Results: Actual vs Predicted Closing Prices
# -----------------------
plt.figure(figsize=(14, 7))
plt.plot(test.index, y_test, label="Actual Close", color="blue", linewidth=2)
plt.plot(
    test.index,
    y_pred,
    label="Predicted Close",
    color="red",
    linestyle="--",
    linewidth=2,
)
plt.title("Actual vs Predicted Closing Prices (Test Set: 2021-2024)")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# -----------------------
# Plot the Daily PnL
# -----------------------
plt.figure(figsize=(14, 7))
plt.plot(results.index, results["pnl"], label="Daily PnL", color="green", linewidth=2)
plt.title("Daily PnL Over Test Period (2021-2024)")
plt.xlabel("Date")
plt.ylabel("PnL (Points)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Performance Metrics & Equity Curve Calculation ---

# Calculate daily dollar PnL, assuming each point is worth $2 on one micro contract
results["dollar_pnl"] = results["pnl"] * 2

# Calculate the equity curve starting from an initial account of $5000
initial_equity = 5000
results["equity"] = initial_equity + results["dollar_pnl"].cumsum()

# Calculate daily equity returns (percentage change)
results["eq_return"] = results["equity"].pct_change()

# Plot the Equity Curve
plt.figure(figsize=(14, 7))
plt.plot(
    results.index, results["equity"], label="Equity Curve", color="blue", linewidth=2
)
plt.title("Equity Curve Over Test Period")
plt.xlabel("Date")
plt.ylabel("Equity ($)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Calculate the running maximum of the equity and the drawdown percentage
results["running_max"] = results["equity"].cummax()
results["drawdown"] = (results["running_max"] - results["equity"])

# Plot the Drawdown
plt.figure(figsize=(14, 7))
plt.plot(results.index, results["drawdown"], label="Drawdown", color="red", linewidth=2)
plt.title("Drawdown Over Test Period")
plt.xlabel("Date")
plt.ylabel("Drawdown ($)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Performance Metrics Calculation ---

# Define the daily risk-free rate for an annual 7% risk-free rate (assume 252 trading days)
rf_daily = 0.07 / 252

# Remove initial NaN from equity returns and compute excess returns (daily return minus risk-free rate)
daily_returns = results["eq_return"].dropna()
excess_returns = daily_returns - rf_daily

# Annualized Sharpe Ratio: sqrt(252) * (mean excess return / std dev of excess returns)
sharpe_ratio = np.sqrt(252) * excess_returns.mean() / excess_returns.std()

# Sortino Ratio: use only the downside (negative) excess returns
downside_returns = excess_returns[excess_returns < 0]
sortino_ratio = (
    np.sqrt(252) * excess_returns.mean() / downside_returns.std()
    if downside_returns.std() != 0
    else np.nan
)

# Total Return on the $5000 account as a percentage
total_return_percentage = ((results["equity"].iloc[-1] / initial_equity) - 1) * 100

# Annualized return: geometric return based on the number of days over the test period
total_days = (results.index[-1] - results.index[0]).days
annualized_return = (results["equity"].iloc[-1] / initial_equity) ** (
    252 / total_days
) - 1

# Calmar Ratio: Annualized return divided by maximum drawdown (max drawdown in percentage)
max_drawdown = results["drawdown"].max()
calmar_ratio = annualized_return / max_drawdown if max_drawdown != 0 else np.nan

# Print performance metrics
print("Performance Metrics:")
print("Total Return on $5000 account: {:.2f}%".format(total_return_percentage))
print("Annualized Return: {:.2f}%".format(annualized_return * 100))
print("Sharpe Ratio: {:.2f}".format(sharpe_ratio))
print("Sortino Ratio: {:.2f}".format(sortino_ratio))
print("Calmar Ratio: {:.2f}".format(calmar_ratio))
