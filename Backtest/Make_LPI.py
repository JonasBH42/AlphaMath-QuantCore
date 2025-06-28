import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("C:/Users/User/Desktop/Projects/Backtest/Csvs/NQ1!_LOG_1D.csv")
main = pd.read_csv("C:/Users/User/Desktop/Projects/Backtest/Csvs/NQ1!_MAIN_1D.csv")


def scale_to_range(series):
        min_val = series.min()
        max_val = series.max()
        scaled = (series - min_val) / (max_val - min_val) * 100
        return scaled

# main["LPI"] = np.log(abs(main["volume"]) / abs(main["close"].pct_change()))

# scaled_df["LPI"] = np.log(scaled_df["volume"] / abs(scaled_df["close"] - scaled_df["close"].shift(1)).replace(0, np.nan)).diff()
# main["LPI"] = np.log(main["volume"] / abs(main["close"] - main["close"].shift(1))) 
data['Return'] = data['Close'] - data['Open']

data.to_csv("C:/Users/User/Desktop/Projects/Backtest/Csvs/NQ1!_LOG_1D.csv", index=False)
# plt.figure(figsize=(10, 6))
# plt.plot(main["ts_event"], main["LPI"], label="LPI")
# plt.xlabel("Date")
# plt.ylabel("LPI")
# plt.title("Liquidity Price Indicator (LPI) Over Time")
# plt.legend()
# plt.show()
