import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = 'Arial Unicode MS'
matplotlib.rcParams['axes.unicode_minus'] = False

from plot_distribution import *

print("loading data...")
df = pd.read_csv('../data/data.txt', sep='\t')

plot_columns = [
    '書名',
    '著者名'
]

for column in plot_columns:
    plot_distribution(column, df)