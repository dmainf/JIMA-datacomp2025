import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

df = pd.read_parquet('data/df_for.parquet')

book_totals = df.groupby('書名', observed=True)['POS販売冊数'].sum().sort_values()
quantiles = book_totals.quantile([0, 0.25, 0.5, 0.75, 1.0])

selected_books = []
for q in [0, 0.25, 0.5, 0.75, 1.0]:
    target = quantiles[q]
    closest_book = (book_totals - target).abs().idxmin()
    selected_books.append(closest_book)

fig, axes = plt.subplots(5, 1, figsize=(14, 12), sharex=True)
fig.suptitle('POS販売冊数の偏差値（その日までの累積統計を使用）', fontsize=16, fontname='Hiragino Sans')

for idx, book_name in enumerate(selected_books):
    book_data = df[df['書名'] == book_name].sort_values('日付').copy()

    book_data['cumulative_mean'] = book_data['POS販売冊数'].expanding().mean()
    book_data['cumulative_std'] = book_data['POS販売冊数'].expanding().std()

    book_data['z_score'] = (book_data['POS販売冊数'] - book_data['cumulative_mean']) / book_data['cumulative_std']
    book_data['deviation_score'] = 50 + 10 * book_data['z_score']

    ax = axes[idx]
    ax.plot(book_data['日付'], book_data['deviation_score'], linewidth=0.8, alpha=0.7)
    ax.axhline(y=50, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax.axhline(y=60, color='orange', linestyle=':', linewidth=0.8, alpha=0.5)
    ax.axhline(y=40, color='orange', linestyle=':', linewidth=0.8, alpha=0.5)

    quantile_label = ['min', '25%', '50%', '75%', 'max'][idx]
    total_sales = book_totals[book_name]
    ax.set_ylabel(f'{quantile_label}\n偏差値', fontname='Hiragino Sans')
    ax.set_title(f'{book_name} (合計: {total_sales:,}冊)', fontsize=10, fontname='Hiragino Sans', loc='left')
    ax.grid(True, alpha=0.3)

    ax.set_ylim(0, 100)

axes[-1].set_xlabel('日付', fontname='Hiragino Sans')
plt.tight_layout()
plt.savefig('pos_sales_deviation_scores.png', dpi=150, bbox_inches='tight')
print('グラフを pos_sales_deviation_scores.png に保存しました')
plt.show()
