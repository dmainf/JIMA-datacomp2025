import pandas as pd
from lib.prepro import *

print('loading data...')
df = pd.read_parquet('data/data.parquet')
print('complete!')
print('normalize title...')
df = normalize_title(df)
print('complete!')

df = remove_volume_number(df, remove_series=False)

frequency_df = df.groupby(['書名']).size().reset_index(name='出現回数')
frequency_df = frequency_df.sort_values('出現回数', ascending=False)

print(f"総レコード数: {len(frequency_df)}")
print(f"\n上位20件:")
print(frequency_df.head(20).to_string(index=False))

frequency_df.to_csv('book_frequency.csv', index=False)
print(f"save /book_frequency.csv")