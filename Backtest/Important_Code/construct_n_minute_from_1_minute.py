import pandas as pd
import numpy as np
from tqdm import tqdm

df2 = pd.read_csv('C:/Users/User/Desktop/AlphaMath-QuantCore/Backtest/Csvs/NQ1!_MAIN_1M_NYC.csv')

df2["datetime"] = pd.to_datetime(df2["datetime"], errors='coerce')
# Create empty DataFrame for 15-minute data
min_15 = pd.DataFrame(columns=['datetime', 'open', 'low', 'high', 'close', 'volume'])

# Process data in chunks of 15 rows
# Group by 15-minute intervals based on datetime

df2['time_group'] = df2['datetime'].apply(
    lambda x: x.replace(minute=(x.minute // 15) * 15, second=0, microsecond=0)
)

for group_time, group_data in tqdm(df2.groupby('time_group'), desc="Processing 15-minute intervals"):
    new_row = {
        'datetime': group_time,
        'open': group_data.iloc[0]['open'],
        'low': group_data['low'].min(),
        'high': group_data['high'].max(),
        'close': group_data.iloc[-1]['close'],
        'volume': group_data['volume'].sum()
    }
    min_15 = pd.concat([min_15, pd.DataFrame([new_row])], ignore_index=True)

# Save to CSV
min_15.to_csv('C:/Users/User/Desktop/AlphaMath-QuantCore/Backtest/Csvs/NQ1!_MAIN_15M.csv', index=False)
