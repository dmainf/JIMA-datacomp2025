import pandas as pd
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from lib.plot_dist import plot_sum_by_book
from multiprocessing import Pool, cpu_count

def process_store(i):
    """各書店のデータを処理する関数"""
    file_path = f'data/df_{i}.parquet'

    if not os.path.exists(file_path):
        return f"Store {i}: File not found, skipping..."

    try:
        df = pd.read_parquet(file_path)
        plot_sum_by_book(df, output_dir='figure', store_id=i)
        return f"Store {i}: Completed ({len(df)} rows)"
    except Exception as e:
        return f"Store {i}: Error - {str(e)}"

if __name__ == '__main__':
    print(f"Using {cpu_count()} CPU cores for parallel processing")
    print(f"\n{'='*60}")
    print("Starting parallel processing...")
    print('='*60)

    # 並列処理で各書店を処理
    with Pool(processes=cpu_count()) as pool:
        results = pool.map(process_store, range(1, 36))

    # 結果を表示
    print(f"\n{'='*60}")
    print("Processing Results:")
    print('='*60)
    for result in results:
        print(result)

    print(f"\n{'='*60}")
    print("All stores processed successfully!")
    print('='*60)