import pandas as pd
import numpy as np
import sys
import gc
sys.path.append('../../')
from lib.prepro import *

MIN_SALES_THRESHOLD = 366

print("=== Loading Data ===")
df_raw = pd.read_parquet('../../data/df.parquet')
df_raw = df_raw[df_raw['POS販売冊数'] >= 0].copy()
df_raw['日付'] = pd.to_datetime(df_raw['日付'].str[:10])
df_raw = remove_volume(df_raw)
cols = df_raw.select_dtypes(include=['object']).columns
df_raw[cols] = df_raw[cols].astype('category')
print("Complete!")

print("=== Extracting Valid Books ===")
book_sales = df_raw.groupby('書名', observed=False)['POS販売冊数'].sum().reset_index()
book_sales.columns = ['書名', 'total_sales']
valid_books = book_sales[book_sales['total_sales'] >= MIN_SALES_THRESHOLD]['書名'].tolist()
print(f"Books with total sales >= {MIN_SALES_THRESHOLD}: {len(valid_books):,}")
print("Complete!")

base_cols = ['日付', '書名']
book_cols = ['出版社','著者名','大分類','中分類','小分類','本体価格']

print("=== Processing book attributes ===")
df_attrs = df_raw[['書名'] + book_cols].copy()
by_book = df_attrs.groupby('書名', observed=False)[book_cols].agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else np.nan).reset_index()
del df_attrs
gc.collect()
print("Complete!")

print("=== Aggregating Sales ===")
df_sales = df_raw.groupby(base_cols, observed=False)['POS販売冊数'].sum().reset_index()
del df_raw
gc.collect()
print("Complete!")

print("=== Creating Full Index ===")
date_range = pd.date_range('2024-01-01', '2024-12-31', freq='D')
valid_books_df = pd.DataFrame({'書名': valid_books, 'key': 0})
dates_df = pd.DataFrame({'日付': date_range, 'key': 0})
df_fullindex = valid_books_df.merge(dates_df, on='key').drop('key', axis=1)
print(f"Full index size: {len(df_fullindex):,}")
print("Complete!")

print("=== Merging Sales Data ===")
df_for = df_fullindex.merge(df_sales, on=base_cols, how='left')
df_for['POS販売冊数'] = df_for['POS販売冊数'].fillna(0).astype(np.int32)
del df_fullindex, df_sales
gc.collect()
print("Complete!")

print("=== Merging Book Attributes ===")
df_for = df_for.merge(by_book, on='書名', how='left')
if '著者名' in df_for.columns:
    if df_for['著者名'].dtype.name == 'category':
        if 'UNKNOWN' not in df_for['著者名'].cat.categories:
            df_for['著者名'] = df_for['著者名'].cat.add_categories(['UNKNOWN'])
    df_for['著者名'] = df_for['著者名'].fillna('UNKNOWN')
print(f"Shape after merging: {df_for.shape}")
print("Complete!")

print("=== Processing Data Types ===")
int_cols = df_for.select_dtypes(include=['int64']).columns.tolist()
float_cols = df_for.select_dtypes(include=['float64']).columns.tolist()
df_for[int_cols] = df_for[int_cols].astype(np.int32)
df_for[float_cols] = df_for[float_cols].astype(np.float32)
categorical_cols = ['書名','出版社','著者名','大分類','中分類','小分類']
for col in categorical_cols:
    if col in df_for.columns:
        df_for[col] = df_for[col].astype('category')

print("=== Removing Unused Categories ===")
for col in categorical_cols:
    if col in df_for.columns and df_for[col].dtype.name == 'category':
        before = df_for[col].cat.categories.size
        df_for[col] = df_for[col].cat.remove_unused_categories()
        after = df_for[col].cat.categories.size
        print(f"{col}: {before} -> {after}")
print("Complete!")

df_for.to_parquet('df_for.parquet', index=False)
print("Saved DataFrame to 'df_for.parquet'")

print("\n=== Data Types ===")
print(df_for.dtypes)
print("\n=== Data Shape ===")
print(df_for.shape)

print("\n=== POS Sales Statistics ===")
print(f"Mean: {df_for['POS販売冊数'].mean():.4f}")
print(f"Variance: {df_for['POS販売冊数'].var():.4f}")
print(f"Std Dev: {df_for['POS販売冊数'].std():.4f}")