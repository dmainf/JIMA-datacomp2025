import pandas as pd
from pathlib import Path

# --- 設定 ---
BOOK_TITLE_COLUMN = '書名'
PRICE_COLUMN = '本体価格'
# 💡 部分検索用の設定に変更 💡
CATEGORY_COLUMN = '大分類'
TARGET_CATEGORY_PARTIAL = 'コミッ' 

DATA_DIR = Path('inconsistent_price_data') # 不一致データExcelがあるディレクトリ
OUTPUT_DIR = Path('price_analysis_results_comic') # 💡 出力ディレクトリを部分検索用に変更

# 出力ディレクトリを作成
OUTPUT_DIR.mkdir(exist_ok=True)

print(f"全店舗の本体価格不一致データから、大分類が'{TARGET_CATEGORY_PARTIAL}'を部分的に含むデータの詳細分析を開始します...")
print("-" * 70)

# inconsistent_price_data フォルダ内の全てのExcelファイルを検索
analysis_files = list(DATA_DIR.glob('inconsistent_body_price_*.xlsx'))

if not analysis_files:
    print(f"エラー: {DATA_DIR} 内に分析対象のExcelファイルが見つかりませんでした。")
else:
    for file_path in analysis_files:
        try:
            store_code = file_path.stem.split('_')[-1]
            print(f"✅ 処理開始: Store {store_code} ({file_path.name})")

            # 不一致データファイルを読み込み
            df = pd.read_excel(file_path)
            
            # 必要な列の存在チェック (念のため)
            if CATEGORY_COLUMN not in df.columns:
                print(f" ⚠️ 警告: Store {store_code} のデータに'{CATEGORY_COLUMN}'カラムがありません。スキップします。")
                continue

            # 💡 フィルタリング処理を部分検索に変更 💡
            # 大分類が「コミッ」を部分的に含むもののみに絞り込み
            df_comic_partial = df[
                df[CATEGORY_COLUMN].astype(str).str.contains(TARGET_CATEGORY_PARTIAL, na=False)
            ].copy()
            
            if df_comic_partial.empty:
                print(f"   情報: Store {store_code} の不一致データに'{TARGET_CATEGORY_PARTIAL}'を含む大分類は見つかりませんでした。")
                continue

            # 書名と本体価格の組み合わせでグループ化し、その件数（出現回数）をカウント
            price_counts = (
                df_comic_partial.groupby([BOOK_TITLE_COLUMN, PRICE_COLUMN]) 
                .size() 
                .reset_index(name='出現回数')
            )
            
            # 書名ごとに本体価格が何種類あるかをカウント
            price_variations = (
                df_comic_partial.groupby(BOOK_TITLE_COLUMN)[PRICE_COLUMN]
                .nunique()
                .reset_index(name='価格種類数')
            )
            
            # 2つの結果を結合
            df_result = pd.merge(price_counts, price_variations, on=BOOK_TITLE_COLUMN)

            # 見やすいようにソート
            df_result = df_result.sort_values(
                by=['価格種類数', BOOK_TITLE_COLUMN, PRICE_COLUMN], 
                ascending=[False, True, True]
            )
            
            # 結果を店舗ごとのExcelファイルに保存
            output_path = OUTPUT_DIR / f'comic_partial_analysis_store_{store_code}.xlsx'
            df_result.to_excel(output_path, index=False)
            
            print(f"   完了。詳細分析結果 ({len(df_comic_partial)}行を分析) を {output_path} に保存しました。")
            
        except Exception as e:
            print(f" ❌ エラーが発生しました (Store {store_code}): {e}")

print("-" * 70)
print(f"全店舗の'{TARGET_CATEGORY_PARTIAL}'分析が完了しました。結果は '{OUTPUT_DIR.name}' フォルダに保存されています。")