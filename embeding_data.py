import pandas as pd
from lib.prepro import *
from lib.fueture_eng import *

print("loading data...")
df_raw = pd.read_csv('data/data.txt', sep='\t')
print("complete!")
df = df_raw.copy()
emb2vec_columns = [
    '書名',
    '著者名'
]
df = emb2vec(df, emb2vec_columns)

print("saving embedded data...")
df.to_parquet('data/embed_data.parquet',
            engine='pyarrow',
            compression='snappy',
            index=False)
print("saved to data/embed_data.parquet")
