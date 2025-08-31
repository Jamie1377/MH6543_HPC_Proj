import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import talib
from concurrent.futures import ProcessPoolExecutor

def calculate_features(df):
    """Calculate technical indicators for a stock dataframe"""
    # Make sure df is sorted by date
    df = df.sort_index()
    
    # Basic price and volume features
    df['Returns'] = df['Close'].pct_change()
    df['Volume_Change'] = df['Volume'].pct_change()
    df['High_Low_Range'] = (df['High'] - df['Low']) / df['Close']
    df['Open_Close_Range'] = (df['Open'] - df['Close']) / df['Close']
    
    # Moving averages
    for window in [5, 10, 20, 50, 100, 200]:
        df[f'MA_{window}'] = df['Close'].rolling(window=window).mean()
        df[f'MA_{window}_Change'] = df[f'MA_{window}'].pct_change()
        df[f'Close_MA_{window}_Ratio'] = df['Close'] / df[f'MA_{window}']
    
    # Volatility measures
    for window in [5, 10, 20, 50]:
        df[f'Volatility_{window}'] = df['Returns'].rolling(window=window).std()
    
    # Price momentum
    for window in [5, 10, 20, 50]:
        df[f'Momentum_{window}'] = df['Close'] / df['Close'].shift(window) - 1
    
    # Technical indicators using TA-Lib
    # RSI
    df['RSI_14'] = talib.RSI(df['Close'].values, timeperiod=14)
    
    # MACD
    macd, macd_signal, macd_hist = talib.MACD(
        df['Close'].values, 
        fastperiod=12, 
        slowperiod=26, 
        signalperiod=9
    )
    df['MACD'] = macd
    df['MACD_Signal'] = macd_signal
    df['MACD_Hist'] = macd_hist
    
    # Bollinger Bands
    upper, middle, lower = talib.BBANDS(
        df['Close'].values, 
        timeperiod=20
    )
    df['BB_Upper'] = upper
    df['BB_Middle'] = middle
    df['BB_Lower'] = lower
    df['BB_Width'] = (upper - lower) / middle
    
    # Stochastic oscillator
    slowk, slowd = talib.STOCH(
        df['High'].values, 
        df['Low'].values, 
        df['Close'].values,
        fastk_period=14,
        slowk_period=3,
        slowk_matype=0,
        slowd_period=3,
        slowd_matype=0
    )
    df['SlowK'] = slowk
    df['SlowD'] = slowd
    
    # Average Directional Index
    df['ADX'] = talib.ADX(
        df['High'].values, 
        df['Low'].values, 
        df['Close'].values, 
        timeperiod=14
    )
    
    # Create target variable (next day's return)
    df['Target_Return'] = df['Returns'].shift(-1)
    df['Target_Direction'] = np.where(df['Target_Return'] > 0, 1, 0)
    
    # Create day of week, month, year features
    df['Day_of_Week'] = df.index.dayofweek
    df['Month'] = df.index.month
    df['Year'] = df.index.year
    
    # Replace inf/-inf with NaN, then drop NaN values
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    
    return df

def process_stock_file(file_path):
    """Process a single stock file"""
    try:
        # Read CSV, skip first 3 rows, set correct column names, parse dates
        col_names = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume', 'Symbol']
        df = pd.read_csv(file_path, skiprows=3, names=col_names, index_col=0, parse_dates=True)
        # Convert price and volume columns to numeric (coerce errors to NaN)
        for col in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Ensure index is DatetimeIndex for dayofweek feature
        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                df.index = pd.to_datetime(df.index, errors='coerce')
            except Exception as e:
                print(f"Could not convert index to datetime for {file_path}: {e}")
                return None, 0
        if df.index.hasnans:
            print(f"NaT values found in index after conversion for {file_path}, skipping file.")
            return None, 0

            # Ensure index is DatetimeIndex for dayofweek feature
            if not isinstance(df.index, pd.DatetimeIndex):
                try:
                    df.index = pd.to_datetime(df.index, errors='coerce')
                except Exception as e:
                    print(f"Could not convert index to datetime for {file_path}: {e}")
                    return None
            if df.index.hasnans:
                print(f"NaT values found in index after conversion for {file_path}, skipping file.")
                return None
        symbol = os.path.basename(file_path).replace('.csv', '')
        # Calculate features
        df_with_features = calculate_features(df)
        # Save processed file
        output_path = f'processed_data/{symbol}_processed.csv'
        df_with_features.to_csv(output_path)
        return output_path, len(df_with_features)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None, 0

def main():
    # Create directory for processed data
    if not os.path.exists('processed_data'):
        os.makedirs('processed_data')
    
    # Get all stock files
    stock_files = [f'stock_data/{f}' for f in os.listdir('stock_data') if f.endswith('.csv')]
    
    print(f"Processing {len(stock_files)} stock files...")
    
    # Process files in parallel
    results = []
    total_rows = 0
    
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        for result, rows in tqdm(executor.map(process_stock_file, stock_files), total=len(stock_files)):
            if result:
                results.append(result)
                total_rows += rows
    
    print(f"Successfully processed {len(results)} files")
    print(f"Total rows in dataset: {total_rows}")
    
    # Create a metadata file with the list of processed files
    with open('processed_files.txt', 'w') as f:
        for result in results:
            f.write(f"{result}\n")

if __name__ == "__main__":
    main()