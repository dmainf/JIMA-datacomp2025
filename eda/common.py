"""
Common utilities for time series forecasting
"""
import pandas as pd
from pathlib import Path
import pickle

# Global constants
BASE_DATE = pd.Timestamp("2024-01-01")

def load_all_stores(data='by_store', exclude_stores=[26, 27]):
    data_path = Path('data') / data
    all_data = []
    for file in sorted(data_path.glob('df_*.parquet')):
        df = pd.read_parquet(file)
        all_data.append(df)
    combined = pd.concat(all_data, ignore_index=True)
    if exclude_stores:
        combined = combined[~combined['書店コード'].isin(exclude_stores)]
    return combined


def save_encoders(encoders, filepath='encoders.pkl'):
    """Save label encoders to file"""
    with open(filepath, 'wb') as f:
        pickle.dump(encoders, f)


def load_encoders(filepath='encoders.pkl'):
    """Load label encoders from file"""
    with open(filepath, 'rb') as f:
        return pickle.load(f)
