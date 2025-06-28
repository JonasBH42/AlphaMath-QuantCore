import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

scaler = StandardScaler()

# Paths
model_path = "C:/Users/User/Desktop/Projects/Backtest/lasso_model.joblib"
csv_file = "C:/Users/User/Desktop/Projects/Backtest/Csvs/NQ1!_1D_7_JAN_FORWARD.csv"

# Load data
data = pd.read_csv(csv_file, parse_dates=["ts_event"], dayfirst=True)
# Rename date column if needed
if "ts_event" in data.columns:
    data.rename(columns={"ts_event": "Date"}, inplace=True)

# Ensure sorted by date
data.sort_values("Date", inplace=True)

# Set Date as index
data.set_index("Date", inplace=True)

# 1-3. Exponential moving averages
data["ema10"] = data["close"].ewm(span=10, adjust=False).mean()
data["ema20"] = data["close"].ewm(span=20, adjust=False).mean()
data["ema50"] = data["close"].ewm(span=50, adjust=False).mean()

# 4. Simple daily return (close - open)
data["return"] = data["close"] - data["open"]

# 5-6. Lagged returns
data["lag1_return"] = data["return"].shift(1)
data["lag2_return"] = data["return"].shift(2)

# 7-8. Rolling std dev of returns
data["roll_std_10"] = data["return"].rolling(window=10).std()
data["roll_std_20"] = data["return"].rolling(window=20).std()

# 9. Lagged close
data["lag1_close"] = data["close"].shift(1)

# Drop rows with NaNs from feature creation
data.dropna(inplace=True)

# Load trained Lasso model
model = joblib.load(model_path)

# Features list
feature_cols = [
    "ema10", "ema20", "ema50", "lag1_return", "lag2_return",
    "roll_std_10", "roll_std_20", "lag1_close"
]

X_scaled = scaler.fit_transform(data[feature_cols])
# Generate predictions
data["prediction"] = model.predict(X_scaled)

# Build results DataFrame, including high/low
results = pd.DataFrame({
    "prediction": data["prediction"],
    "open": data["open"],
    "high": data["high"],
    "low": data["low"],
    "return": data["return"]
}, index=data.index)

# Backtest PnL calculation with stop-loss logic
mask_long = results["prediction"] > results["open"]
mask_short = ~mask_long
# Adverse moves
# mask_stop_long = (results["open"] - results["low"]) >= 400
# mask_stop_short = (results["high"] - results["open"]) >= 400
# Base PnL by direction
pnl = np.where(mask_long, results["return"], -results["return"])
# Override stops to fixed -400 loss
# pnl = np.where(mask_long & mask_stop_long, -400, pnl)
# pnl = np.where(mask_short & mask_stop_short, -400, pnl)

results["pnl"] = pnl
results["success"] = results["pnl"] > 0

# Metrics
total_pnl = results["pnl"].sum()
num_wins = results["success"].sum()
num_losses = (~results["success"]).sum()
winners = results.loc[results["success"], "pnl"]
losers = results.loc[~results["success"], "pnl"]
avg_winner = winners.mean() if not winners.empty else float('nan')
avg_loser = losers.mean() if not losers.empty else float('nan')
risk_reward = avg_winner / abs(avg_loser) if avg_loser != 0 else float('inf')
# Daily Sharpe ratio (annualized by sqrt(252))
daily_mean = results["pnl"].mean()
daily_std = results["pnl"].std()
sharpe_ratio = (daily_mean / daily_std) * np.sqrt(252) if daily_std != 0 else float('nan')

# Print metrics
print(f"Total PnL: {total_pnl:.2f}")
print(f"Number of Wins: {num_wins}")
print(f"Number of Losses: {num_losses}")
print(f"Average Winner PnL: {avg_winner:.4f}")
print(f"Average Loser PnL: {avg_loser:.4f}")
print(f"Risk/Reward Ratio: {risk_reward:.4f}")
print(f"Sharpe Ratio (annualized): {sharpe_ratio:.4f}")

# Plot daily PnL
plt.figure(figsize=(12, 6))
results['pnl'].plot(title='Daily PnL')
plt.xlabel('Date')
plt.ylabel('PnL')
plt.tight_layout()
plt.show()

# Plot cumulative PnL over time
plt.figure(figsize=(12, 6))
results['pnl'].cumsum().plot(title='Cumulative PnL Over Time')
plt.xlabel('Date')
plt.ylabel('Cumulative PnL')
plt.tight_layout()
plt.show()
