import yfinance as yf
import pandas as pd

def yahoo_csv_maker(ticker, start_date, end_date):

    data = yf.download(ticker, start=start_date, end=end_date)
    
    data.columns = [col[0].lower() for col in data.columns]
    data.reset_index(inplace=True)
    data = data.rename(columns={"Date": "ts_event"})
    data['ts_event'] = pd.to_datetime(data['ts_event']).dt.strftime('%Y-%m-%dT%H:%M:%S')
    
    data.to_csv(f"C:/Users/User/Desktop/Projects/Backtest/Csvs/{ticker}.csv", index=False)




yahoo_csv_maker("^IRX", "2014-01-01", "2025-01-07")