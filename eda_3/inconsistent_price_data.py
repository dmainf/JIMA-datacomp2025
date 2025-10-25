import pandas as pd
from pathlib import Path

# --- 設定（実際のデータに合わせて列名を調整してください） ---
BOOK_TITLE_COLUMN = '書名'
PRICE_COLUMN = '本体価格'
DATA_DIR = Path('data')
OUTPUT_DIR = Path('inconsistent_price_data')

# 出力ディレクトリを作成
OUTPUT_DIR.mkdir(exist_ok=True)

print("書名が同じで本体価格が異なるデータをチェックします...")
print("-" * 50)

# 全店舗のファイルを処理
store_files = list(DATA_DIR.glob('df_*.parquet'))

if not store_files:
    print(f"エラー: {DATA_DIR} 内に 'df_*.parquet' ファイルが見つかりませんでした。")
else:
    inconsistent_data_found = False

    for file_path in store_files:
        try:
            store_code = file_path.stem.split('_')[-1]
            df = pd.read_parquet(file_path)

            # 必要な列の存在チェック
            required_columns = [BOOK_TITLE_COLUMN, PRICE_COLUMN]
            if not all(col in df.columns for col in required_columns):
                print(f" ⚠️ 警告 (Store {store_code}): 必要な列 {required_columns} がありません。スキップします。")
                continue
            
            # 書名ごとに本体価格のユニーク数（種類数）をカウント
            price_variation = df.groupby(BOOK_TITLE_COLUMN)[PRICE_COLUMN].nunique()
            inconsistent_titles = price_variation[price_variation > 1].index.tolist()

            if inconsistent_titles:
                inconsistent_data_found = True
                print(f"✅ Store {store_code}: 本体価格が異なる {len(inconsistent_titles)} 種類の書名が見つかりました。")

                # 該当する書名を含む全レコードを抽出
                df_inconsistent = df[df[BOOK_TITLE_COLUMN].isin(inconsistent_titles)].copy()
                
                # --- 💡 ここでソート処理を追加 💡 ---
                # 書名カラムで昇順にソートします
                df_inconsistent.sort_values(by=BOOK_TITLE_COLUMN, inplace=True)
                
                # 結果をファイルに保存
                output_path = OUTPUT_DIR / f'inconsistent_body_price_{store_code}.xlsx'
                
                # openpyxlのインストールが必要なため、エラーが出ていた場合は以下を実行してください
                # pip install openpyxl
                df_inconsistent.to_excel(output_path, index=False)
                print(f"   詳細データを {output_path} にソート済みで保存しました。")
            else:
                print(f"   Store {store_code}: 本体価格の不一致は見つかりませんでした。")

        except Exception as e:
            # openpyxlエラーを再度表示する可能性があるため、注意を促します
            if "'openpyxl'" in str(e):
                 print(f" ❌ エラーが発生しました (Store {store_code}): No module named 'openpyxl'")
                 print("    💡 Excel出力には `pip install openpyxl` が必要です。")
            else:
                print(f" ❌ エラーが発生しました (Store {store_code}): {e}")

    print("-" * 50)
    if inconsistent_data_found:
        print(f"処理完了。本体価格の不一致データは '{OUTPUT_DIR.name}' フォルダにソート済みで保存されています。")
    else:
        print("全ての店舗で、書名に対する本体価格の不一致は見つかりませんでした。")