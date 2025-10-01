import pandas as pd
from lib.prepro import clean_df

df = pd.read_csv('data/data.txt', sep='\t')
df = clean_df(df)

# データの基本情報
print("=== データの形状 ===")
print(df.shape)

print("\n=== カラム名 ===")
print(df.columns.tolist())

print("\n=== 欠損値の数 ===")
print(df.isnull().sum())

print("\n=== 欠損値の割合 (%) ===")
print((df.isnull().sum() / len(df) * 100).round(2))

print("\n=== 最初の5行 ===")
print(df.head())

print("\n=== データ型 ===")
print(df.dtypes)

print("\n=== カラム削除後 ===")
print(f"データの形状: {df.shape}")
print(f"カラム名: {df.columns.tolist()}")