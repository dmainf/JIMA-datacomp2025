import pandas as pd
import numpy as np
from collections import Counter
import sys
sys.path.append('../../')
from lib.prepro import *

print("=== Loading Data ===")
df = pd.read_parquet('../../data/df.parquet')
print("Complete!")

print("Dropping Unnecessary Columns and Cleaning Volume Data...")
delete_cols = [
    '月','日','累積日数','ISBN','書店名','住所',
    'メッシュ1','メッシュ2','メッシュ3','メッシュ4','メッシュ5','メッシュ6','メッシュ7','メッシュ8','メッシュ9',
    '周辺合計人口','合計人口','駅距離',
    '市区別_夜間人口','市区別_昼間人口','昼間フラグ','昼間/夜間割合'
]
df = df.drop(columns=delete_cols, errors='ignore')
df = remove_volume(df)
print("Complete!")

print("Processing Data Types...")
int_cols = [
    '書店コード','駅構内','複合施設','独立店舗'
]
int_cols = df.select_dtypes(include=['int64']).columns.tolist()
float_cols = df.select_dtypes(include=['float64']).columns.tolist()
df[int_cols] = df[int_cols].astype(np.int32)
df[float_cols] = df[float_cols].astype(np.float32)
print("Complete!")

print("Grouping Data by Store, Book, and Date...")
group_keys = ['書店コード', '書名', '日付']
res_df = df.groupby(group_keys)['POS販売冊数'].sum().reset_index()
mode_cols = [c for c in df.columns if c not in group_keys + ['POS販売冊数']]
df_keys = df[group_keys].astype(str)
df['_group_id'], groups = pd.factorize(df_keys.apply(lambda x: '_'.join(x), axis=1))
for col in mode_cols:
    original_dtype = df[col].dtype
    grouped_vals = np.split(df[col].to_numpy(), np.cumsum(np.bincount(df['_group_id']))[:-1])
    mode_vals = []
    for vals in grouped_vals:
        uniques = set(vals)
        if len(uniques) == 1:
            mode_vals.append(vals[0])
        else:
            counts = Counter(vals)
            max_count = max(counts.values())
            mode_val = next(k for k, v in counts.items() if v == max_count)
            mode_vals.append(mode_val)
    res_df[col] = mode_vals
    if np.issubdtype(original_dtype, np.number):
        res_df[col] = res_df[col].astype(original_dtype)
    print(f"Processed mode for column: {col}")
print("Complete!")

res_df.to_parquet('sales_df.parquet', index=False)
print("Saved grouped DataFrame to 'groupby_df.parquet'")

print("\n=== データ型 ===")
print(res_df.dtypes)
