import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = 'Arial Unicode MS'
matplotlib.rcParams['axes.unicode_minus'] = False
import sys
sys.path.append('../lib')
from plot_dist import *
from prepro import clean_df

print("loading data...")
df_raw = pd.read_csv('../data/data.txt', sep='\t')

df = clean_df(df_raw)
#元データ
plot_columns = [
    '書名',
    '著者名'
]
for column in plot_columns:
    plot_distribution(column, df_raw)

#count_encodingしたあとのグラフ
plot_count_columns = [
    '書名',
    '著者名'
]
for column in plot_count_columns:
    plot_count_distribution(column, df)