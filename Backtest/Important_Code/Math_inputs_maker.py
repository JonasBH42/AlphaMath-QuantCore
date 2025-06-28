import pandas as pd
import numpy as np

# Sample DataFrame structure
df = pd.read_csv("C:/Users/User/Desktop/Projects/Backtest/Csvs/NQ1!_SCALED_1D.csv")

# Add mathematical inputs as new columns

# 1. Price-Action Volatility Transforms (PVT)
n = 5  # lookback period
df["PVT"] = df["close"].diff().abs().rolling(n).sum() / n

# 2. Trend-Consistency Index (TCI)
df["TCI"] = df["close"].diff().apply(np.sign).rolling(n).mean()

# 3. Momentum Entropy (ME)
def calculate_entropy(window):
    counts = window.value_counts(normalize=True)
    entropy = -np.sum(counts * np.log2(counts + 1e-9))  # avoid log(0)
    return entropy

df["ME"] = df["close"].diff().rolling(n).apply(calculate_entropy, raw=False)

# 4. Price-Acceleration Factor (PAF)
df["PAF"] = (df["close"].diff() - df["close"].shift(1).diff()) / df["close"].shift(2)

# 5. Cross-Sector Influence Score (CSI)
# For simplicity, assume cross-sector index simulation with the same values (normally would use sector returns)
df["CSI"] = (df["close"] / df["close"].shift(1)).rolling(n).mean()

# 6. Liquidity-Pressure Index (LPI)
epsilon = 1e-9  # to avoid division by zero
df["LPI"] = df["volume"] / (df["close"].diff().abs() + epsilon)

# 7. Adaptive Fourier Frequency Component (AFFC)
def simplified_fourier_transform(series, freq=0.1):
    t = np.arange(len(series))
    omega = 2 * np.pi * freq
    return np.sum(series * np.cos(omega * t))

df["AFFC"] = df["close"].rolling(n).apply(lambda x: simplified_fourier_transform(x), raw=True)

# 8. Cumulative Reversion Index (CRI)
df["mean_close"] = df["close"].rolling(n).mean()
df["std_close"] = df["close"].rolling(n).std()
df["CRI"] = ((df["close"] - df["mean_close"]) / (df["std_close"] + epsilon)).rolling(n).sum()

# 9. Market-Response Indicator (MRI)
# Here, assume a simple market index proxy (average of 'high' and 'low')
df["market_index"] = (df["high"] + df["low"]) / 2
df["MRI"] = df["close"].diff() / df["market_index"].diff()

# 10. Nonlinear Momentum Interactions (NMI)
df["NMI"] = df["close"].diff() * np.sign(df["close"].shift(2).diff())

# Drop intermediary columns to keep only final features
df.drop(columns=["mean_close", "std_close", "market_index"], inplace=True)

# Display the updated dataframe
df.to_csv("C:/Users/User/Desktop/Projects/Backtest/Csvs/Math_csvs/NQ1!_1D_MATH_PARAMS.csv", index=False)
