import pandas as pd
import numpy as np
import yfinance as yf
from tqdm import tqdm
import os
import time

# Create directory for data
if not os.path.exists('stock_data'):
    os.makedirs('stock_data')

def download_sp500_symbols():
    """Get list of S&P 500 symbols from Wikipedia using requests to avoid HTTP 403"""
    import requests
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    headers = {'User-Agent': 'Mozilla/5.0 (compatible; S&P500Fetcher/1.0)'}
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    tables = pd.read_html(response.text)
    sp500_table = tables[0]
    symbols = sp500_table['Symbol'].tolist()
    return symbols

def download_stock_data(symbol, start_date='2000-01-01', end_date='2025-01-01'):
    """Download historical data for a given stock symbol"""
    try:
        # Download data
        data = yf.download(symbol, start=start_date, end=end_date, progress=False)
        
        # Skip if no data or very little data
        if len(data) < 100:
            return None
            
        # Add symbol column
        data['Symbol'] = symbol
        
        return data
    except Exception as e:
        print(f"Error downloading {symbol}: {e}")
        return None

def main():
    # Get S&P 500 symbols
    print("Getting S&P 500 symbols...")
    symbols = download_sp500_symbols()
    
    print(f"Found {len(symbols)} symbols. Starting download...")
    
    # Set date range for 20+ years of data
    start_date = '2000-01-01'
    end_date = '2025-01-01'  # Using future date to get all data up to present
    
    # Download data for each symbol
    successful_downloads = 0
    
    for symbol in tqdm(symbols):
        data = download_stock_data(symbol, start_date, end_date)
        
        if data is not None:
            # Save to CSV
            data.to_csv(f'stock_data/{symbol}.csv')
            successful_downloads += 1
            
        # Add small delay to prevent rate limiting
        time.sleep(0.5)
    
    print(f"Successfully downloaded data for {successful_downloads} symbols")

if __name__ == "__main__":
    main()