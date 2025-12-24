import pandas as pd
from lib import *

print("loading data...")
normalized_title_list = pd.read_csv('normalized_title_list.csv')
print("Complete!")

print_df(normalized_title_list)

df = normalized_title_list[['normalized_title', '出現回数']].copy()
df = df.rename(columns={'normalized_title': '書名'})

df = remove_volume_number(df)

result = df.groupby('書名', as_index=False)['出現回数'].sum()
result = result.rename(columns={'書名': 'normalized_title'})

print("saving to normalized_title(remove).csv...")
result.to_csv('normalized_title(remove).csv', index=False)
print("Complete!")

print_df(result)
