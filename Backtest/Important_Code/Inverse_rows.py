import pandas as pd



def inverse_column_order(path):
    df = pd.read_csv(path)
    df = df.iloc[::-1].reset_index(drop=True) 
    df.to_csv(path, index=False)

path = "C:/Users/User/Desktop/Projects/Backtest/Csvs/US10Y.csv"
inverse_column_order(path)