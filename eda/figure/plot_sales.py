import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from multiprocessing import Pool, cpu_count
from functools import partial
import time

plt.rcParams['font.family'] = 'Hiragino Sans'
plt.rcParams['axes.unicode_minus'] = False

def process_store(i):
    file_path = f'../data/by_store/df_{i}.parquet'

    if not os.path.exists(file_path):
        return f"Store {i}: File not found, skipped"

    df = pd.read_parquet(file_path)

    df['売上'] = df['本体価格'] * df['POS販売冊数']

    grouped = df.groupby(['書名', '累積日数'], observed=True)['売上'].sum().reset_index()

    grouped['累積売上'] = grouped.groupby('書名', observed=True)['売上'].cumsum()

    final_sales = grouped.groupby('書名', observed=True)['累積売上'].last().sort_values(ascending=False)
    top10_books = final_sales.head(10)
    top10_book_names = set(top10_books.index.tolist())

    fig, ax = plt.subplots(figsize=(16, 10))

    other_books = grouped[~grouped['書名'].isin(top10_book_names)]

    if len(other_books) > 0:
        other_books_grouped = other_books.groupby('書名', observed=True)
        for book_name, book_data in other_books_grouped:
            ax.plot(book_data['累積日数'].values, book_data['累積売上'].values,
                    alpha=0.3, linewidth=0.5, color='gray', rasterized=True)

    top10_grouped = grouped[grouped['書名'].isin(top10_book_names)].groupby('書名', observed=True)
    for book_name, book_data in top10_grouped:
        ax.plot(book_data['累積日数'].values, book_data['累積売上'].values,
                label=str(book_name), alpha=0.8, linewidth=2)

    ax.set_xlabel('累積日数', fontsize=12)
    ax.set_ylabel('累積売上（円）', fontsize=12)
    title = f'書名ごとの累積売上推移（上位10位を強調表示） - 書店{i}'
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.tight_layout()

    output_file = f'cumulative_sales_store_{i}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close(fig)

    top10_list = '\n'.join([f"  {j}. {book}: {sales:,.0f}円"
                            for j, (book, sales) in enumerate(top10_books.items(), 1)])

    return f"Store {i}: Processed {len(df)} rows, {len(final_sales)} unique books\nTop 10:\n{top10_list}"

if __name__ == '__main__':
    print(f"{'='*60}")
    print(f"Starting parallel processing with {cpu_count()} CPU cores")
    print(f"{'='*60}\n")

    start_time = time.time()

    store_ids = list(range(1, 36))

    with Pool(processes=cpu_count()) as pool:
        results = pool.map(process_store, store_ids)

    for result in results:
        print(result)
        print(f"{'='*60}\n")

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"{'='*60}")
    print(f"All stores processed successfully!")
    print(f"Total time: {elapsed_time:.2f} seconds")
    print(f"Average time per store: {elapsed_time/35:.2f} seconds")
    print(f"{'='*60}")
