import pandas as pd
from datetime import datetime

def change_date_format(csv_name):
    path = f"C:/Users/User/Desktop/Projects/Backtest/Csvs/{csv_name}.csv"
    
    # Read the CSV file
    df = pd.read_csv(path)
    
    # Convert the 'ts_event' column to ISO format
    # df['ts_event'] = pd.to_datetime(df['ts_event']).dt.strftime("%Y-%d-%m %H:%M:%S")
    df['ts_event'] = pd.to_datetime(df['ts_event'], utc=True, dayfirst=True).dt.strftime("%Y-%m-%d %H:%M:%S+00:00")

    
    # Save the transformed DataFrame to a new CSV file
    df.to_csv(path, index=False)

# Example usage
change_date_format('test')