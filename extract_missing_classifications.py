import pandas as pd
from lib.prepro import clean_df

print("loading data...")
df = pd.read_parquet('data/embed_data.parquet')
print("complete!")

print("cleaning df...")
df_cleaned = clean_df(df)
print("complete!")

# 大分類、中分類、小分類が同時に欠損しているレコードを抽出
print("\nextracting records with missing classifications...")
missing_all_classifications = df_cleaned[
    df_cleaned['大分類'].isna() &
    df_cleaned['中分類'].isna() &
    df_cleaned['小分類'].isna()
]

print(f"found {len(missing_all_classifications)} records with all classifications missing")
print(f"shape: {missing_all_classifications.shape}")

print("\nsaving to data/missing_classifications.parquet...")
missing_all_classifications.to_parquet('data/missing_classifications.parquet')
print("complete!")
