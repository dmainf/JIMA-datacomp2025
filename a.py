import pandas as pd
from lib.prepro import clean_df

df_raw = pd.read_csv('data/data.txt', sep='\t')
df = clean_df(df_raw)

df_original = pd.DataFrame(df)

df_grouped = df_original.groupby(['大分類', '中分類', '小分類']).size().reset_index(name='小分類出現数')
fig = px.sunburst(
    df_grouped,
    path=['大分類', '中分類', '小分類'],
    values='小分類出現数',
    title='大分類・中分類・小分類の階層構造（サンバーストチャート）'
)
fig.show()

# 3. ツリーマップを作成する場合は、以下のコードを使用できます。
fig_treemap = px.treemap(
    df_grouped,
    path=['大分類', '中分類', '小分類'],
    values='小分類出現数',
    title='大分類・中分類・小分類の階層構造（ツリーマップ）'
)
fig_treemap.show()

print("=== データの形状 ===")
print(df_raw.shape)

print("\n=== 欠損値の数 ===")
print(df_raw.isnull().sum())

print("\n=== 欠損値の割合 (%) ===")
print((df_raw.isnull().sum() / len(df_raw) * 100).round(2))


# データの基本情報
print("=== データの形状 ===")
print(df.shape)

print("\n=== 欠損値の数 ===")
print(df.isnull().sum())

print("\n=== 欠損値の割合 (%) ===")
print((df.isnull().sum() / len(df) * 100).round(2))

print("\n=== データ型 ===")
print(df.dtypes)

print(df.head())