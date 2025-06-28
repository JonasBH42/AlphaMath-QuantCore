import requests
import pandas as pd
from datetime import datetime

# URL for CNN Fear & Greed API (unofficial endpoint)
url = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"

# Fake headers to make the request look like a normal browser
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
    "Accept": "application/json",
    "Referer": "https://edition.cnn.com/",
    "Origin": "https://edition.cnn.com"
}

def get_fear_greed_data():
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch data: {response.status_code}")
    
    data = response.json()
    
    # Extract historical data
    history = data['fear_and_greed_historical']['data']
    
    # Convert to DataFrame
    df = pd.DataFrame(history)
    
    # Convert timestamps to readable dates
    df['date'] = pd.to_datetime(df['x'], unit='ms')
    
    # Rename columns
    df.rename(columns={'y': 'fear_greed_index'}, inplace=True)
    
    # Select and reorder columns
    df = df[['date', 'fear_greed_index']]
    
    return df

if __name__ == "__main__":
    df = get_fear_greed_data()
    print(df)
    # Optionally save to CSV
    df.to_csv("fear_greed_history.csv", index=False)
