import pandas as pd
from lib.prepro import clean_df

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

print("loading data...")
df_raw = pd.read_csv('data/data.txt', sep='\t')
df = clean_df(df_raw)

print("=== データの形状 ===")
print(df_raw.shape)

print("\n=== 欠損値の数 ===")
print(df_raw.isnull().sum())

print("\n=== 欠損値の割合 (%) ===")
print((df_raw.isnull().sum() / len(df_raw) * 100).round(2))

print("\n=== データ型 ===")
print(df_raw.dtypes)


# データの基本情報
print("=== データの形状 ===")
print(df.shape)

print("\n=== 欠損値の数 ===")
print(df.isnull().sum())

print("\n=== 欠損値の割合 (%) ===")
print((df.isnull().sum() / len(df) * 100).round(2))

print("\n=== データ型 ===")
print(df.dtypes)