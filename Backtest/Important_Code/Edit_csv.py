import pandas as pd

# Load the CSV file into a DataFrame
file_path = 'C:/Users/User/Desktop/Projects/Backtest/Csvs/NQ1!_MAIN_1H.csv'
df = pd.read_csv(file_path)

# Convert 'ts_event' column to datetime
df['ts_event'] = pd.to_datetime(df['ts_event'])

# Filter rows where 'ts_event' is after 00:00 on 2024-09-25
filtered_df = df[df['ts_event'] > '2024-09-25 00:00:00']

# Display the filtered DataFrame
# Save the filtered DataFrame to a new CSV file
filtered_df.to_csv('C:/Users/User/Desktop/Projects/Backtest/Csvs/filtered_NQ1!_1H.csv', index=False)