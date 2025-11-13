import pandas as pd
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
from lib.prepro import *

print("loading data...")
df = pd.read_parquet('../../data/df.parquet')
df = remove_volume_number(df, remove_series=False)
print(f"complete!")

df['売上'] = df['POS販売冊数'] * df['本体価格']
output_dir = Path('./by_store')
output_dir.mkdir(parents=True, exist_ok=True)
other_cols = [col for col in df.columns
            if col not in [
                '書店コード',
                '月',
                '日',
                '累積日数',
                '日付',
                '書名',
                'POS販売冊数',
                '本体価格',
                '売上'
            ]]
agg_dict = {
    'POS販売冊数': 'sum',
    '本体価格': 'mean',
    '売上': 'sum'
}
for col in other_cols:
    agg_dict[col] = 'first'

print("processing stores...")
for store_code, df_store in df.groupby('書店コード'):
    if len(df_store) > 0:
        df_grouped = df_store.groupby([
            '月',
            '日',
            '累積日数',
            '日付',
            '書名'
        ], as_index=False).agg(agg_dict)
        df_grouped['書店コード'] = store_code
        output_path = output_dir / f'df_{store_code}.parquet'
        df_grouped.to_parquet(output_path, index=False)
        print(f"Store {store_code}: {len(df_grouped)} rows (grouped from {len(df_store)} rows)")

print("Complete!")