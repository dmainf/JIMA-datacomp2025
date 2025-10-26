import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = 'Arial Unicode MS'
matplotlib.rcParams['axes.unicode_minus'] = False
import os

def plot_distribution(column, df=None):
    """指定されたカラムの分布をプロットする"""
    if df is None:
        print("loading data...")
        df = pd.read_csv('../data/data.txt', sep='\t')

    print(f"総行数: {len(df)}")
    print(f"ユニークな{column}数: {df[column].nunique()}")
    null_count = df[column].isnull().sum()
    null_percentage = (null_count/len(df[column])*100).round(2)
    print(f"欠損値の数: {null_count}")
    print(f"欠損値の割合: {null_percentage}")

    counts = df[column].value_counts().head(20)

    plt.figure(figsize=(14, 9))
    # 数値型と文字列型の両方に対応
    if counts.index.dtype == 'object':
        names = [str(name).replace('\u3000', ' ').strip() for name in counts.index]
    else:
        names = [str(name) for name in counts.index]
    plt.barh(range(len(counts)), counts.values)
    plt.yticks(range(len(counts)), names, fontsize=10)
    plt.xlabel('出現回数', fontsize=11)
    plt.ylabel(column, fontsize=11)
    plt.title(f'{column}の分布(上位20件)', fontsize=13)
    plt.gca().invert_yaxis()

    # 欠損値情報をグラフに表示
    plt.text(0.98, 0.98, f'欠損値: {null_count:,}件 ({null_percentage}%)',
             transform=plt.gca().transAxes,
             fontsize=10,
             verticalalignment='top',
             horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    print(f"\n上位20件の{column}:")
    print(counts)

    os.makedirs('figure', exist_ok=True)
    output_file = f'figure/{column}_dist.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nグラフを {output_file} に保存しました")

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        columns = sys.argv[1:]
        for column in columns:
            plot_distribution(column)
    else:
        print("使用方法: python plot_dist.py [カラム名1] [カラム名2] ...")


def plot_count_distribution(column, df=None, log_x=False, log_y=False):
    """指定されたカラムの出現回数ごとの頻度分布をプロットする"""
    if df is None:
        print("loading data...")
        df = pd.read_csv('../data/data.txt', sep='\t')
    # 元データから出現回数を計算
    value_counts = df[column].value_counts()
    # 欠損値をカウント
    null_count = df[column].isnull().sum()
    total_count = len(df)
    null_percentage = (null_count / total_count * 100).round(2)

    # 出現回数ごとの頻度を計算
    counts_of_counts = value_counts.value_counts().sort_index()

    plt.figure(figsize=(20, 9))
    plt.bar(counts_of_counts.index, counts_of_counts.values)

    # x軸ラベル
    if log_x:
        plt.xlabel(f'各{column}の出現回数（対数スケール）', fontsize=11)
        plt.xscale('log')
    else:
        plt.xlabel(f'各{column}の出現回数', fontsize=11)

    # y軸ラベル
    if log_y:
        plt.ylabel(f'該当する{column}の種類数（対数スケール）', fontsize=11)
        plt.yscale('log')
    else:
        plt.ylabel(f'該当する{column}の種類数', fontsize=11)

    plt.title(f'{column}の出現回数ごとの頻度分布', fontsize=13)
    plt.grid(True, alpha=0.3, which='both')

    # 欠損値情報をグラフに表示
    plt.text(0.98, 0.98, f'欠損値: {null_count:,}件 ({null_percentage}%)',
             transform=plt.gca().transAxes,
             fontsize=10,
             verticalalignment='top',
             horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    print(f"\n出現回数ごとの{column}種類数:")
    print(counts_of_counts)

    os.makedirs('figure', exist_ok=True)
    log_suffix = ''
    if log_x and log_y:
        log_suffix = '_loglog'
    elif log_x:
        log_suffix = '_logx'
    elif log_y:
        log_suffix = '_logy'
    output_file = f'figure/{column}_counts_dist{log_suffix}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nグラフを {output_file} に保存しました")

if __name__ == '__main__':
    if len(sys.argv) > 1:
        columns = sys.argv[1:]
        for column in columns:
            plot_count_distribution(column)
    else:
        print("使用方法: python plot_counts_dist.py [カラム名1] [カラム名2] ...")


def plot_sum_by_book(df=None, output_dir='figure', store_id=None):
    """書名ごとの累積売上を折れ線グラフで表示"""
    if df is None:
        print("loading data...")
        df = pd.read_parquet('../data/data.parquet')

    print("売上を計算中...")
    df['売上'] = df['本体価格'] * df['POS販売冊数']

    print("書名と累積日数でグループ化中...")
    grouped = df.groupby(['書名', '累積日数'], observed=True)['売上'].sum().reset_index()

    print("累積売上を計算中...")
    grouped['累積売上'] = grouped.groupby('書名', observed=True)['売上'].cumsum()

    print("最終売上をソート中...")
    final_sales = grouped.groupby('書名', observed=True)['累積売上'].last().sort_values(ascending=False)
    top10_books = final_sales.head(10)
    top10_book_names = set(top10_books.index.tolist())
    print("上位10位の書名:")
    for i, (book, sales) in enumerate(top10_books.items(), 1):
        print(f"{i}. {book}: {sales:,.0f}円")
    print(f"\n書名のユニーク数: {len(final_sales)}")

    print("グラフを作成中...")
    fig, ax = plt.subplots(figsize=(16, 10))

    # その他の書名データを分離
    other_books = grouped[~grouped['書名'].isin(top10_book_names)]

    # その他の書名を一括プロット（最適化版）
    if len(other_books) > 0:
        print(f"その他の書名をプロット中（{len(other_books['書名'].unique())}件）...")
        # グループごとにデータを準備して一括プロット
        other_books_grouped = other_books.groupby('書名', observed=True)
        for book_name, book_data in other_books_grouped:
            ax.plot(book_data['累積日数'].values, book_data['累積売上'].values,
                    alpha=0.3, linewidth=0.5, color='gray', rasterized=True)

    print("上位10位の書名をプロット中...")
    top10_grouped = grouped[grouped['書名'].isin(top10_book_names)].groupby('書名', observed=True)
    for book_name, book_data in top10_grouped:
        ax.plot(book_data['累積日数'].values, book_data['累積売上'].values,
                label=str(book_name), alpha=0.8, linewidth=2)

    ax.set_xlabel('累積日数', fontsize=12)
    ax.set_ylabel('累積売上（円）', fontsize=12)
    title = '書名ごとの累積売上推移（上位10位を強調表示）'
    if store_id is not None:
        title += f' - 書店{store_id}'
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    if store_id is not None:
        output_file = os.path.join(output_dir, f'cumulative_sales_store_{store_id}.png')
    else:
        output_file = os.path.join(output_dir, 'cumulative_sales_by_book.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close(fig)  # メモリ解放
    print(f"\nグラフを {output_file} に保存しました")
    print(f"書名数: {len(final_sales)}")
