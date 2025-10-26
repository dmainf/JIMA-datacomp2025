import pandas as pd
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from lib.plot_dist import *
from lib.prepro import *

for i in range(1, 36):
    print(f"\n{'='*60}")
    print(f"Processing df_{i}.parquet...")
    print('='*60)

    file_path = f'data/df_{i}.parquet'

    if not os.path.exists(file_path):
        print(f"File {file_path} not found, skipping...")
        continue

    print("loading data...")
    df = pd.read_parquet(file_path)
    print(f"complete! Loaded {len(df)} rows")

    plot_sum_by_book(df, output_dir='figure', store_id=i)

print(f"\n{'='*60}")
print("All stores processed successfully!")
print('='*60)

df = remove_volume_number(df)
plot_sum_by_book(df)