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
    print(f"欠損値の数: {df[column].isnull().sum()}")
    print(f"欠損値の割合: {(df[column].isnull().sum()/len(df[column])*100).round(2)}")

    counts = df[column].value_counts().head(20)

    plt.figure(figsize=(14, 9))
    names = [name.replace('\u3000', ' ').strip() for name in counts.index]
    plt.barh(range(len(counts)), counts.values)
    plt.yticks(range(len(counts)), names, fontsize=10)
    plt.xlabel('出現回数', fontsize=11)
    plt.ylabel(column, fontsize=11)
    plt.title(f'{column}の分布(上位20件)', fontsize=13)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    print(f"\n上位20件の{column}:")
    print(counts)

    os.makedirs('figure', exist_ok=True)
    output_file = f'figure/{column}_distribution.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nグラフを {output_file} に保存しました")

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        columns = sys.argv[1:]
        for column in columns:
            plot_column_distribution(column)
    else:
        print("使用方法: python plot_distribution.py [カラム名1] [カラム名2] ...")
