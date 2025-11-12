import pandas as pd
from lib.prepro import *

print('loading data...')
df = pd.read_parquet('data/data.parquet')
print('complete!')
df = normalize_title(df)
print('complete!')

df['書名'] = df['書名'].apply(lambda x: x.rsplit('_', 1)[0] if pd.notna(x) and '_' in str(x) else x)
print('書名の最も右の_以降を削除完了')

frequency_df = df.groupby(['書名']).size().reset_index(name='出現回数')
frequency_df = frequency_df.sort_values('出現回数', ascending=False)

print(f"総レコード数: {len(frequency_df)}")
print(f"\n上位20件:")
print(frequency_df.head(20).to_string(index=False))

frequency_df.to_csv('book_frequency.csv', index=False)
print(f"結果を book_frequency.csv に保存しました")
