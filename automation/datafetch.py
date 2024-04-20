import os
import json
import yfinance as yf
from datetime import datetime

def update_with_today(token, ASSETS, TODAYS, START_DATE):
    """
    token -> Yahoo Finance Recognized token.

    Fetching Stock Data:
        1. Goes to data/assets/
        2. Fetches the {token}.csv file.
        3. Finds the date from which this has to be updated.
        {token}.csv
        4. Fetches is using yfinance.
        5. Replaces old file.

    Fetching Opens:
        1. Finds today()
        2. Fetches today's open.
        3. Replaces it in OPENS.
        {token}.json
    """

    # Fetching Stock Data
    data_path = os.path.join(ASSETS, f"{token}.csv")
    
    # Fetch data
    data = yf.download(token, start=START_DATE)

    # Save updated data
    data.to_csv(data_path)

    # Fetching Opens, Highs, Lows, and Closes
    today_data = yf.Ticker(token).history(period="1d")
    today_open = today_data['Open'].values[0]
    today_high = today_data['High'].values[0]
    today_low = today_data['Low'].values[0]
    today_close = today_data['Close'].values[0]

    # Save today's trading data to JSON file
    today_trading_data = {
        'open': today_open,
        'high': today_high,
        'low': today_low,
        'close': today_close
    }
    open_path = os.path.join(TODAYS, f"{token}.json")
    with open(open_path, 'w') as json_file:
        json.dump(today_trading_data, json_file)

    print(f"Updated {token} data and today's open.")

if __name__ == '__main__':
    ASSETS = "../data/assets/"
    TODAYS = "../data/todays/"

    TOKENS = ['GBTC', 'ETCG', 'ETHE', 'GDLC']
    START_DATE = datetime(2019, 11, 22)

    for token in TOKENS:
        update_with_today(token, ASSETS, TODAYS, START_DATE)
