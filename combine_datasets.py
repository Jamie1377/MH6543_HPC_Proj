import pandas as pd
import numpy as np
import os
from tqdm import tqdm

def main():
    # Read the list of processed files
    with open('processed_files.txt', 'r') as f:
        processed_files = [line.strip() for line in f.readlines()]
    
    print(f"Combining {len(processed_files)} processed files...")
    
    # Create directory for combined data
    if not os.path.exists('combined_data'):
        os.makedirs('combined_data')
    
    # Process in chunks to avoid memory issues
    chunk_size = 20
    chunks = [processed_files[i:i + chunk_size] for i in range(0, len(processed_files), chunk_size)]
    
    combined_chunks = []
    
    # Process each chunk
    for i, chunk in enumerate(tqdm(chunks)):
        dfs = []
        
        for file_path in chunk:
            try:
                df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                dfs.append(df)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
        
        # Combine chunk
        if dfs:
            combined_chunk = pd.concat(dfs)
            chunk_file = f'combined_data/chunk_{i}.csv'
            combined_chunk.to_csv(chunk_file)
            combined_chunks.append(chunk_file)
    
    print(f"Created {len(combined_chunks)} chunks")
    print("Now combining all chunks into one large dataset...")
    
    # Combine all chunks
    combined_df = pd.DataFrame()
    
    for chunk_file in tqdm(combined_chunks):
        chunk = pd.read_csv(chunk_file, index_col=0, parse_dates=True)
        combined_df = pd.concat([combined_df, chunk])
    
    # Save final combined dataset
    combined_df.to_csv('combined_data/full_dataset.csv')
    
    print(f"Created combined dataset with {len(combined_df)} rows")

if __name__ == "__main__":
    main()