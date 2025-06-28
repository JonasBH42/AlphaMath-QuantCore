import pandas as pd
import gc
from tqdm import tqdm
import math
import numpy as np


def calculate_rsi(data, window=14):

    delta = data["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_atr(data, window=14):

    # Calculate True Range (TR)
    data["high-low"] = data["high"] - data["low"]
    data["high-close"] = abs(data["high"] - data["close"].shift(1))
    data["low-close"] = abs(data["low"] - data["close"].shift(1))

    data["tr"] = data[["high-low", "high-close", "low-close"]].max(axis=1)

    # Calculate ATR
    data["ATR"] = data["tr"].rolling(window=window).mean()

    # Remove the intermediate columns
    data.drop(columns=["high-low", "high-close", "low-close", "tr"], inplace=True)

    return data["ATR"]


def calculate_ema(data, length=30):
    # Calculate the Exponential Moving Average (EMA) for a given DataFrame.
    ema = data["close"].ewm(span=length, adjust=False).mean()
    return ema


def calculate_dema(data, length=30):
    # Calculate the Exponential Moving Average (EMA)
    ema = data["close"].ewm(span=length, adjust=False).mean()
    # Calculate the Double Exponential Moving Average (DEMA)
    dema = 2 * ema - ema.ewm(span=length, adjust=False).mean()
    return dema


def calculate_tema(data, length=30):
    # Calculate the Triple Exponential Moving Average (TEMA) for a given DataFrame.
    ema = data["close"].ewm(span=length, adjust=False).mean()
    ema2 = ema.ewm(span=length, adjust=False).mean()
    ema3 = ema2.ewm(span=length, adjust=False).mean()
    tema = 3 * (ema - ema2) + ema3
    return tema


def squeeze_momentum_indicator(
    data, length=20, mult=2.0, lengthKC=20, multKC=1.5, use_true_range=True
):
    # Calculate Bollinger Bands
    source = data["close"]
    basis = source.rolling(window=length).mean()
    dev = mult * source.rolling(window=length).std()
    upperBB = basis + dev
    lowerBB = basis - dev

    # Calculate Keltner Channels
    ma = source.rolling(window=lengthKC).mean()
    if use_true_range:
        true_range = np.maximum(
            data["high"] - data["low"],
            np.maximum(
                abs(data["high"] - data["close"].shift(1)),
                abs(data["low"] - data["close"].shift(1)),
            ),
        )
        range_ma = true_range.rolling(window=lengthKC).mean()
    else:
        range_ma = (data["high"] - data["low"]).rolling(window=lengthKC).mean()

    upperKC = ma + range_ma * multKC
    lowerKC = ma - range_ma * multKC

    # Squeeze conditions
    sqzOn = (lowerBB > lowerKC) & (upperBB < upperKC)
    sqzOff = (lowerBB < lowerKC) & (upperBB > upperKC)
    noSqz = ~(sqzOn | sqzOff)

    # Linear regression calculation (approximated with rolling mean for simplicity)
    avg_high_low = (
        data["high"].rolling(window=lengthKC).max()
        + data["low"].rolling(window=lengthKC).min()
    ) / 2
    avg_close = source.rolling(window=lengthKC).mean()
    val = (source - (avg_high_low + avg_close) / 2).rolling(window=lengthKC).mean()

    # # Bar colors
    # bcolor = np.where(val > 0,
    #                   np.where(val > val.shift(1), 'lime', 'green'),
    #                   np.where(val < val.shift(1), 'red', 'maroon'))

    # # Squeeze colors
    # scolor = np.where(noSqz, 'blue', np.where(sqzOn, 'black', 'gray'))

    # Adding calculated data to the DataFrame
    data["val"] = val
    # data['bcolor'] = bcolor
    # data['scolor'] = scolor
    data["sqzOn"] = sqzOn
    data["sqzOff"] = sqzOff
    data["noSqz"] = noSqz

    return data


def moving_average(src, length, ma_type="sma"):
    """Calculate the moving average."""
    if len(src) < length:
        raise ValueError("Not enough data points to calculate moving average")

    if ma_type == "sma":
        # Simple Moving Average
        return np.mean(src[-length:])
    elif ma_type == "ema":
        # Exponential Moving Average
        weights = np.exp(np.linspace(-1.0, 0.0, length))
        weights /= weights.sum()
        return np.dot(src[-length:], weights)
    else:
        raise ValueError("Unsupported moving average type. Use 'sma' or 'ema'.")


def calculate_stdev(src, length):
    """Calculate the standard deviation."""
    if len(src) < length:
        raise ValueError("Not enough data points to calculate standard deviation")
    avg = np.mean(src[-length:])
    squared_deviations = [(x - avg) ** 2 for x in src[-length:]]
    return np.sqrt(np.mean(squared_deviations))


def bollinger_bands(src, length=20, mult=2, ma_type="sma"):
    """Calculate Bollinger Bands."""
    # Basis line: moving average
    basis = moving_average(src, length, ma_type)

    # Deviation: standard deviation
    dev = mult * calculate_stdev(src, length)

    # Upper and lower bands
    upper = basis + dev
    lower = basis - dev

    return basis, upper, lower


def plot_bollinger_bands(df):
    for i in tqdm(range(0, len(df) - 1), total=len(df) - 1):
        if i < 20:
            continue
        basis, upper, lower = bollinger_bands(
            df.iloc[i - 20 : i]["close"], length=20, mult=2, ma_type="sma"
        )
        df.loc[i, "basis"] = basis
        df.loc[i, "upper"] = upper
        df.loc[i, "lower"] = lower

    return df


file_path = "C:/Users/User/Desktop/Projects/Backtest/Csvs/Math_csvs/NQ1!_1D_MATH_PARAMS_WITH_LAGGED.csv"
data = pd.read_csv(file_path)

# data["ATR"] = calculate_atr(data, window=1)
# data["ATR_ratio"] = data["ATR"] / data["ATR"].shift(1)
# data.drop(columns=["ATR"], inplace=True)

# data["RSI"] = calculate_rsi(data, window=14)

data["TEMA25"] = calculate_tema(data, length=25)
# data["DEMA60"] = calculate_dema(data, length=60)
# data["EMA70"] = calculate_ema(data, length=70)
# data = plot_bollinger_bands(data)
# data = squeeze_momentum_indicator(data)


data.to_csv(
    "C:/Users/User/Desktop/Projects/Backtest/Csvs/Math_csvs/NQ1!_1D_MATH_PARAMS_WITH_LAGGED.csv",
    index=False,
)
# print(data["basis"])
