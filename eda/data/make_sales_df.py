import pandas as pd
import numpy as np
from collections import Counter
import sys
sys.path.append('../../')
from lib.prepro import *

TOP_K = 500

print("=== Loading Data ===")
df_raw = pd.read_parquet('../../data/df.parquet')
print("Complete!")

print("=== Creating Base DataFrame ===")
df = df_raw[['日付','書店コード','書名','POS販売冊数']]
df = remove_volume(df)
print("Complete!")

print("=== Processing Book Attributes ===")
book_cols = ['出版社','著者名','大分類','中分類','小分類','本体価格']
by_book = df_raw.groupby('書名')[book_cols].agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else np.nan).reset_index()
for col in ['出版社','著者名','大分類','中分類','小分類']:
    by_book[col] = by_book[col].astype('category')
print("Complete!")

print("=== Processing Store Attributes with Mesh Convolution ===")
mesh_3th_cols = [f'メッシュ{i}_人口' for i in range(1, 10)]
mesh_4th_cols = [f'4th_メッシュ{i}_人口' for i in range(1, 10)]
store_cols = [
    '書店コード',
    '駅距離',
    '開店時間(平)','閉店時間(平)','営業時間(平)','開店時間(特)','閉店時間(特)','営業時間(特)',
    '駅構内','複合施設','独立店舗',
    '市区別_夜間人口','市区別_昼間人口'
]
by_store = df_raw[store_cols + mesh_3th_cols + mesh_4th_cols].groupby('書店コード').first().reset_index()
pop_3th_by_mesh = by_store[mesh_3th_cols].values.reshape(-1, 3, 3)
pop_4th_by_mesh = by_store[mesh_4th_cols].values.reshape(-1, 3, 3)

def extract_2x2_features(mesh_data):
    features = []
    for i in range(2):
        for j in range(2):
            region = mesh_data[:, i:i+2, j:j+2]
            features.append(region.sum(axis=(1, 2)))
            features.append(region.mean(axis=(1, 2)))
            features.append(region.max(axis=(1, 2)))
            features.append(region.min(axis=(1, 2)))
    return np.column_stack(features)
pop_3th_features = extract_2x2_features(pop_3th_by_mesh)
pop_4th_features = extract_2x2_features(pop_4th_by_mesh)
feature_names_3th = []
feature_names_4th = []
for i in range(2):
    for j in range(2):
        for stat in ['sum', 'mean', 'max', 'min']:
            feature_names_3th.append(f'mesh_3th_{i}{j}_{stat}')
            feature_names_4th.append(f'mesh_4th_{i}{j}_{stat}')
for idx, name in enumerate(feature_names_3th):
    by_store[name] = pop_3th_features[:, idx]
for idx, name in enumerate(feature_names_4th):
    by_store[name] = pop_4th_features[:, idx]

by_store = by_store.drop(columns=mesh_3th_cols + mesh_4th_cols)
print("Complete!")

print("=== Selecting Top Books ===")
top_books = df['書名'].value_counts().head(TOP_K).index.tolist()
print(f"対象書籍数: {len(top_books)}")
print("Complete!")

print("=== Creating Full Index (Date x Store x Book) ===")
all_dates = pd.date_range('2024-01-01', '2024-12-31', freq='D')
all_stores = df['書店コード'].unique()
full_index = pd.MultiIndex.from_product([all_dates, all_stores, top_books], names=['日付','書店コード','書名'])
df_base = pd.DataFrame(index=full_index).reset_index()
print(f"ベース行数: {len(df_base):,}")
print("Complete!")

print("=== Aggregating Sales ===")
df_sales = df.groupby(['日付','書店コード','書名'])['POS販売冊数'].sum().reset_index()
df_final = df_base.merge(df_sales, on=['日付','書店コード','書名'], how='left')
df_final['POS販売冊数'] = df_final['POS販売冊数'].fillna(0).astype(np.int32)
del df_base, df_sales
import gc
gc.collect()
print("Complete!")

print("=== Merging with Store and Book Attributes ===")
df_final = df_final.merge(by_store, on='書店コード', how='left')
print("Merged store attributes")
df_final = df_final.merge(by_book, on='書名', how='left')
print("Merged book attributes")
print("Complete!")

print("=== Processing Data Types ===")
int_cols = df_final.select_dtypes(include=['int64']).columns.tolist()
float_cols = df_final.select_dtypes(include=['float64']).columns.tolist()
df_final[int_cols] = df_final[int_cols].astype(np.int32)
df_final[float_cols] = df_final[float_cols].astype(np.float32)
print("Complete!")

print("=== Converting String Columns to Category ===")
for col in ['出版社','著者名','大分類','中分類','小分類']:
    if col in df_final.columns:
        df_final[col] = df_final[col].astype('category')
        print(f"Converted {col} to category")
print("Complete!")

df_final.to_parquet('sales_df.parquet', index=False)
print("Saved DataFrame to 'sales_df.parquet'")

print("\n=== データ型 ===")
print(df_final.dtypes)
print("\n=== データ形状 ===")
print(df_final.shape)
