import pandas as pd
from lib.prepro import *
from lib.fueture_eng import *

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

print("loading data...")
df_raw = pd.read_csv('data/data.txt', sep='\t')
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

print("###after cleaning###")
print("loading data...")
df = pd.read_parquet('data/embed_data.parquet')
print("complete!")
df = clean_df(df)
onehot_columns = [
    '大分類'
]
df = onehot_enc(df, onehot_columns)

print(df.head())
print("=== データの形状 ===")
print(df.shape)
print("\n=== 欠損値の数 ===")
print(df.isnull().sum())
print("\n=== 欠損値の割合 (%) ===")
print((df.isnull().sum() / len(df) * 100).round(2))
print("\n=== データ型 ===")
print(df.dtypes)

print()