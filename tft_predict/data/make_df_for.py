import pandas as pd
import numpy as np
import gc
from lib import *

def add_time_features(df):
    print("\n=== Adding Time-Series Features ===")

    df['month'] = df['日付'].dt.month
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12).astype(np.float32)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12).astype(np.float32)

    df['dayofweek'] = df['日付'].dt.dayofweek.astype('category')

    df['day'] = df['日付'].dt.day
    df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31).astype(np.float32)
    df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31).astype(np.float32)

    holidays_2024 = [
        '2024-01-01', '2024-01-08', '2024-02-11', '2024-02-12', '2024-02-23',
        '2024-03-20', '2024-04-29', '2024-05-03', '2024-05-04', '2024-05-05',
        '2024-05-06', '2024-07-15', '2024-08-11', '2024-08-12', '2024-09-16',
        '2024-09-22', '2024-09-23', '2024-10-14', '2024-11-03', '2024-11-04',
        '2024-11-23'
    ]
    holidays = pd.to_datetime(holidays_2024)
    df['is_holiday'] = df['日付'].isin(holidays).astype(np.float32)

    df = df.drop(['month', 'day'], axis=1)

    print("Created features:")
    print("  - month_sin, month_cos (Sin/Cos encoding of month)")
    print("  - dayofweek (0=Monday, 6=Sunday)")
    print("  - day_sin, day_cos (Sin/Cos encoding of day)")
    print(f"  - is_holiday ({df['is_holiday'].sum():.0f} holiday days in dataset)")

    return df

print("=== Loading Data ===")
df_raw = pd.read_parquet('df.parquet')
df_raw = df_raw[df_raw['POS販売冊数'] >= 0].copy()
df_raw['日付'] = pd.to_datetime(df_raw['日付'].str[:10])
cols = df_raw.select_dtypes(include=['object']).columns
df_raw[cols] = df_raw[cols].astype('category')
print("Complete!")

base_cols = ['日付', '書名']
book_cols = ['出版社','著者名','大分類','中分類','小分類','本体価格']

print("=== Processing book attributes ===")
df_attrs = df_raw[['書名'] + book_cols].copy()
by_book = df_attrs.groupby('書名', observed=False)[book_cols].agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else np.nan).reset_index()
by_book['書名_base'] = by_book['書名'].astype(str).str.split('_').str[0]
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
valid_books = by_book['書名'].unique()
print(f"Number of unique books: {len(valid_books):,}")
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

df_for = add_time_features(df_for)

print("=== Processing Data Types ===")
int_cols = df_for.select_dtypes(include=['int64']).columns.tolist()
float_cols = df_for.select_dtypes(include=['float64']).columns.tolist()
df_for[int_cols] = df_for[int_cols].astype(np.int32)
df_for[float_cols] = df_for[float_cols].astype(np.float32)
categorical_cols = ['書名','書名_base','出版社','著者名','大分類','中分類','小分類']
for col in categorical_cols:
    if col in df_for.columns:
        df_for[col] = df_for[col].astype('category')
if 'dayofweek' in df_for.columns:
    df_for['dayofweek'] = df_for['dayofweek'].astype('category')
categorical_cols.append('dayofweek')

print("=== Removing Unused Categories ===")
for col in categorical_cols:
    if col in df_for.columns and df_for[col].dtype.name == 'category':
        before = df_for[col].cat.categories.size
        df_for[col] = df_for[col].cat.remove_unused_categories()
        after = df_for[col].cat.categories.size
        print(f"{col}: {before} -> {after}")
print("Complete!")

print_df(df_for)

df_for.to_parquet('df_for.parquet', index=False)
print("Saved DataFrame to 'df_for.parquet'")