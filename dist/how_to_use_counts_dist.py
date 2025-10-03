import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = 'Arial Unicode MS'
matplotlib.rcParams['axes.unicode_minus'] = False
import sys
sys.path.append('..')
from lib.prepro import *

from plot_counts_distribution import *

print("loading data...")
df_raw = pd.read_csv('../data/data.txt', sep='\t')
df = clean_df(df_raw)

plot_columns = [
    '書名',
    '著者名'
]

for column in plot_columns:
    plot_counts_distribution(column, df)
