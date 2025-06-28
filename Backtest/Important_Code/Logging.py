import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


data = pd.read_csv("C:/Users/User/Desktop/Projects/Backtest/Csvs/NQ1!_MAIN_1D.csv")

copy = data.copy()
copy = copy.drop(columns=["volume", "ts_event"])
copy = copy.dropna()
copy = np.log(copy)

copy.insert(0, "ts_event", data["ts_event"])
copy["volume"] = data["volume"]


plt.figure(figsize=(10, 5))
plt.plot(copy.index, copy["close"], label="Close Price")
plt.xlabel("Timestamp")
plt.ylabel("Log Close Price")
plt.title("Log Close Price Over Time")
plt.legend()
plt.grid(True)
plt.show()

copy.to_csv("C:/Users/User/Desktop/Projects/Backtest/Csvs/NQ1!_LOG_1D.csv", index=False)
