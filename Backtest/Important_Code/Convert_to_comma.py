import pandas as pd

# Read the CSV file with spaces as the separator



def convert_to_comma(input_file):
    df = pd.read_csv(f"C:/Users/User/Desktop/Projects/Backtest/Csvs/{input_file}", delim_whitespace=True)
    df.to_csv(f"C:/Users/User/Desktop/Projects/Backtest/Csvs/{input_file}", index=False, sep=',')

convert_to_comma("cpi.csv")