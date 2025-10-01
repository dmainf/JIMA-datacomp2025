import pandas as pd

df_raw = pd.read_csv('data/data.txt', sep='\t')

# データの基本情報
print("=== データの形状 ===")
print(df_raw.shape)

print("\n=== カラム名 ===")
print(df_raw.columns.tolist())

print("\n=== 最初の5行 ===")
print(df_raw.head())

print("\n=== 欠損値の数 ===")
print(df_raw.isnull().sum())

print("\n=== 欠損値の割合 (%) ===")
print((df_raw.isnull().sum() / len(df_raw) * 100).round(2))

print("\n=== データ型 ===")
print(df_raw.dtypes)

print("\n=== 基本統計量 ===")
print(df_raw.describe())

print("\n=== 数値列の統計量 ===")
print(df_raw.describe(include='number'))

print("\n=== カテゴリ列の統計量 ===")
print(df_raw.describe(include='object'))
