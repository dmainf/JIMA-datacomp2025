import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import os
from multiprocessing import Pool, cpu_count
import time
import numpy as np
plt.rcParams['font.family'] = 'Hiragino Sans'
plt.rcParams['axes.unicode_minus'] = False
def process_store(i):
    file_path = f'../data/by_store/df_{i}.parquet'
    if not os.path.exists(file_path):
        return f"Store {i}: File not found, skipped"
    df = pd.read_parquet(file_path)
    grouped = df.groupby(['書名', '累積日数'], observed=True)['売上'].sum().reset_index()
    total_sales = grouped.groupby('書名', observed=True)['売上'].sum()
    top30_book_names_sorted = total_sales.nlargest(30).index.tolist()
    fig, ax = plt.subplots(figsize=(16, 10))
    other_books_df = grouped[~grouped['書名'].isin(top30_book_names_sorted)]
    if not other_books_df.empty:
        segments = [g[['累積日数', '売上']].values for _, g in other_books_df.groupby('書名', observed=True) if len(g) > 1]
        if segments:
            line_collection = LineCollection(segments, color='gray', alpha=0.3, linewidth=0.5, rasterized=True)
            ax.add_collection(line_collection)
            ax.autoscale_view()
    lines = []
    labels = []
    for book_name in top30_book_names_sorted:
        book_data = grouped[grouped['書名'] == book_name]
        if not book_data.empty:
            line, = ax.plot(book_data['累積日数'], book_data['売上'], label=str(book_name), alpha=0.8, linewidth=1.5, rasterized=True)
            lines.append(line)
            labels.append(str(book_name))
    ax.set_xlabel('累積日数', fontsize=12)
    ax.set_ylabel('売上（円）', fontsize=12)
    ax.set_title(f'書名ごとの日次売上推移（上位30位を強調表示） - 書店{i}', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 366)
    ticks = list(ax.get_xticks())
    if 366 not in ticks:
        ticks.append(366)
    ticks = [tick for tick in ticks if tick <= 366]
    ticks.sort()
    ax.set_xticks(ticks)
    all_sales_values = grouped['売上'].values
    min_y = all_sales_values.min()
    max_y = all_sales_values.max()
    y_buffer = (max_y - min_y) * 0.05
    tick_interval = 5000
    start_tick = np.floor((min_y - y_buffer) / tick_interval) * tick_interval
    end_tick = np.ceil((max_y + y_buffer) / tick_interval) * tick_interval
    yticks = np.arange(start_tick, end_tick + tick_interval, tick_interval)
    ax.set_ylim(min_y - y_buffer, max_y + y_buffer)
    ax.set_yticks(yticks)
    if lines:
        ax.legend(lines, labels, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    total_store_sales = df['売上'].sum()
    ax.text(0.98, 0.02, f'合計売上: {total_store_sales:,.0f}円',
            verticalalignment='bottom', horizontalalignment='right',
            transform=ax.transAxes,
            color='black', fontsize=12, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=5))
    plt.tight_layout()
    os.makedirs('daily', exist_ok=True)
    output_file = f'daily/daily_sales_store_{i}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close(fig)
    num_books = len(grouped['書名'].unique())
    return f"Store {i}: Plotted {num_books} books, highlighting top 30 with sorted legend."
if __name__ == '__main__':
    print(f"{'='*60}")
    print(f"Starting parallel processing with {cpu_count()} CPU cores")
    print(f"{'='*60}")
    start_time = time.time()
    store_ids = list(range(1, 36))
    with Pool(processes=cpu_count()) as pool:
        results = pool.map(process_store, store_ids)
    for result in results:
        print(result)
        print(f"{'='*60}")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"{'='*60}")
    print(f"All stores processed successfully!")
    print(f"Total time: {elapsed_time:.2f} seconds")
    print(f"Average time per store: {elapsed_time/len(store_ids):.2f} seconds")
    print(f"{'='*60}")