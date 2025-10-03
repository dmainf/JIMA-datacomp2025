import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = 'Arial Unicode MS'
matplotlib.rcParams['axes.unicode_minus'] = False
import sys
import os
sys.path.append('..')
from lib.prepro import *

def plot_counts_distribution(column, df=None):
    """指定されたカラムの出現回数ごとの頻度分布をプロットする"""
    if df is None:
        print("loading data...")
        df_raw = pd.read_csv('../data/data.txt', sep='\t')
        df = clean_df(df_raw)

    df = df[df[column] > 0]

    counts_of_counts = df[column].value_counts().sort_index()

    plt.figure(figsize=(20, 9))
    plt.bar(counts_of_counts.index, counts_of_counts.values)

    plt.xlabel(f'各{column}の出現回数', fontsize=11)
    plt.ylabel(f'該当する{column}の種類数', fontsize=11)
    plt.title(f'{column}の出現回数ごとの頻度分布', fontsize=13)
    plt.tight_layout()

    print(f"\n出現回数ごとの{column}種類数:")
    print(counts_of_counts)

    os.makedirs('figure', exist_ok=True)
    output_file = f'figure/{column}_counts_distribution.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nグラフを {output_file} に保存しました")

if __name__ == '__main__':
    if len(sys.argv) > 1:
        columns = sys.argv[1:]
        for column in columns:
            plot_counts_distribution(column)
    else:
        print("使用方法: python plot_counts_distribution.py [カラム名1] [カラム名2] ...")
