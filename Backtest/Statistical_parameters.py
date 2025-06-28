import pandas as pd
import numpy as np
import gc

gc.collect()

file_path = "C:/Users/User/Desktop/Projects/Backtest/Csvs/Stats/NQ1_SCALED_1D_with_volatility.csv"
df = pd.read_csv(file_path)

# Ensure proper datetime format
df['ts_event'] = pd.to_datetime(df['ts_event'])

# Calculate required parameters

# Normalized Range: (High - Low) / Close
df['Normalized Range'] = (df['high'] - df['low']) / df['close']

# Squared Returns: (Close_t - Close_(t-1))^2
df['Squared Returns'] = df['close'].diff()**2

# Close-to-Open Returns: (Close_t - Open_t) / Open_t
# df['Close-to-Open Returns'] = np.log(np.clip(df['close'] - df['open'], 1e-8, None)) / df['open']
df['Close-to-Open Returns'] = (df['close'] - df['open']) / df['open']

# Log Returns: log(Close_t / Close_(t-1))
df['Log Returns'] = np.log(df['close'] / df['close'].shift(1))

# Rate of Change (ROC): (Close_t - Close_(t-n)) / Close_(t-n) * 100 (using n=1 for daily)
df['ROC'] = np.log((df['close'].diff() / df['close'].shift(1)) * 100)

# On-Balance Volume (OBV)
df['OBV_diff'] = (df['OBV'] - df['OBV'].min()) / (df['OBV'].max() - df['OBV'].min())
df.drop(columns=['OBV'], inplace=True)


# Volume Rate of Change (Volume ROC):
# df['Volume Rate of Change'] = np.log((df['volume'].diff() / df['volume'].shift(1)) * 100)
vroc = np.log(df['volume'] / df['volume'].shift(1)) * 100
df['Volume Rate of Change'] = (vroc - vroc.min()) / (vroc.max() - vroc.min())

def calculate_normalized_volatility(df, length=30):
    trading_days = 253
    SqrTime = np.sqrt(trading_days / length)

    xMaxC = df['close'].rolling(window=length).max()
    xMinC = df['close'].rolling(window=length).min()
    xMaxH = df['high'].rolling(window=length).max()
    xMinL = df['low'].rolling(window=length).min()

    Vol = ((0.6 * np.log(xMaxC / xMinC) * SqrTime) + (0.6 * np.log(xMaxH / xMinL) * SqrTime)) * 0.5

    # Clamping the values between 0 and 2.99
    nRes = np.clip(Vol, 0, 2.99)

    return nRes

df['normalized_volatility'] = calculate_normalized_volatility(df)

output_file_path = "C:/Users/User/Desktop/Projects/Backtest/Csvs/Stats/NQ1_SCALED_1D_with_volatility_2.csv"
df.to_csv(output_file_path, index=False)