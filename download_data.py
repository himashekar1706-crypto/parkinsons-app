import pandas as pd
import urllib.request
import os

def download_uci_parkinsons():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data"
    output_file = "parkinsons.csv"
    
    print(f"Downloading Parkinson's dataset from {url}...")
    try:
        urllib.request.urlretrieve(url, output_file)
        print(f"Successfully downloaded to {output_file}")
        
        # Verify the file
        df = pd.read_csv(output_file)
        print(f"Dataset shape: {df.shape}")
        print("First few columns:", list(df.columns[:5]))
    except Exception as e:
        print(f"Failed to download or read dataset: {e}")

if __name__ == "__main__":
    download_uci_parkinsons()
