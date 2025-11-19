import pandas as pd
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))
from lib.prepro import remove_volume

print("=== Loading Data ===")
data_path = Path(__file__).resolve().parents[2] / 'data' / 'df.parquet'
df = pd.read_parquet(data_path)
print(f"Total records: {len(df)}")

print("\n=== Calculating Release Dates per Volume ===")
volume_release_dates = df.groupby('書名')['日付'].min().reset_index()
volume_release_dates.columns = ['書名', '発売日']
print(f"Total volumes: {len(volume_release_dates)}")

print("\n=== Filtering books with volume numbers ===")
volume_release_dates = volume_release_dates[volume_release_dates['書名'].str.contains('_', na=False)]
print(f"Books with volume format: {len(volume_release_dates)}")

print("\n=== Extracting Series and Volume Numbers ===")
volume_release_dates['作品_シリーズ'] = volume_release_dates['書名'].str.rsplit('_', n=1).str[0]
volume_release_dates['巻数'] = volume_release_dates['書名'].str.rsplit('_', n=1).str[1]

print("\n=== Sorting volumes by release date ===")
volume_release_dates = volume_release_dates.sort_values('発売日')

print("\n=== Grouping by Series ===")
series_release_dates = volume_release_dates.groupby('作品_シリーズ').agg(
    巻数=('巻数', list),
    発売日=('発売日', lambda x: list(x.dt.date)), # 日までで保存
    シリーズ_最初の発売日=('発売日', 'min') # ソート用に一時的に作成
).reset_index()

# シリーズの最初の発売日順にソートする (古い順)
series_release_dates = series_release_dates.sort_values('シリーズ_最初の発売日').reset_index(drop=True)

print(f"Total series: {len(series_release_dates)}")
print("\nSample release dates (sorted by first release):")
for i in range(min(5, len(series_release_dates))):
    row = series_release_dates.iloc[i]
    print(f"\n{row['作品_シリーズ']}")
    print(f"  巻数: {row['巻数'][:5]}{'...' if len(row['巻数']) > 5 else ''}")
    print(f"  発売日: {row['発売日'][:5]}{'...' if len(row['発売日']) > 5 else ''}")

# === ▼ 修正箇所 ▼ ===
# 保存前にソート用カラムを削除
series_release_dates = series_release_dates.drop(columns=['シリーズ_最初の発売日'])

output_path = Path(__file__).parent / 'release_dates.parquet'
series_release_dates.to_parquet(output_path, index=False)
print(f"\nSaved to: {output_path}")