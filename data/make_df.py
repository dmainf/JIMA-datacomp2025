import pandas as pd
import sys
sys.path.append('..')
from lib.prepro import *

print("loading data...")
df_raw = pd.read_parquet('data.parquet')
store_detail = pd.read_csv('store_detail.csv')
print("complete!")

print("=== データの形状 ===")
print(df_raw.shape)
print("\n=== 欠損値の数 ===")
print(df_raw.isnull().sum())
print("\n=== 欠損値の割合 (%) ===")
print((df_raw.isnull().sum() / len(df_raw) * 100).round(2))
print("\n=== データ型 ===")
print(df_raw.dtypes)
print()
print()

print("###after cleaning###")
df = df_raw.copy()
df = clean_df(df, store_detail)

print("=== データの形状 ===")
print(df.shape)
print("\n=== 欠損値の数 ===")
print(df.isnull().sum())
print("\n=== 欠損値の割合 (%) ===")
print((df.isnull().sum() / len(df) * 100).round(2))
print("\n=== データ型 ===")
print(df.dtypes)
print()

print("saving dataflame to data/df.parquet...")
df.to_parquet('df.parquet')
print("complete!")