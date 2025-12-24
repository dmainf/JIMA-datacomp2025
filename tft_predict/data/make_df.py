import pandas as pd
from lib import *

print("loading data...")
df_raw = pd.read_parquet('data.parquet')
normalized_title_list = pd.read_csv('normalized_title_list.csv')
print("Complete!")

print_df(df_raw)

df = df_raw.copy()
df = drop_unsure(df)
df = drop_unstore(df)
df = filter_by_total_sales(df, min_sales=100)
df = normalize_titles(df, normalized_title_list)
df = remove_volume_number(df)
df = normalize_author(df)

print_df(df)

print("saving dataflame to data/df.parquet...")
df.to_parquet('df.parquet')
print("Complete!")