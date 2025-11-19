import pandas as pd
from lib.prepro import *

print('loading data...')
df = pd.read_parquet('data/data.parquet')
print('complete!\n')
print('normalize title...')
df = normalize_title(df)
df = remove_volume(df)
print('complete!')

frequency_df = df.groupby(['書名']).size().reset_index(name='出現回数')
frequency_df = frequency_df.sort_values('出現回数', ascending=False)

frequency_df.to_csv('book_frequency.csv', index=False)
print(f"save /book_frequency.csv")