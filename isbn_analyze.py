
import pandas as pd
from collections import Counter

file_path = 'data/embed_data.parquet'

# ハイフンパターンのカウンター
total_pattern_counts = Counter()

try:
    # Parquetファイルから'ISBN'カラムのみを読み込む
    df = pd.read_parquet(file_path, columns=['ISBN'])

    # ISBNを文字列に変換し、スペースを削除
    cleaned_isbn = df['ISBN'].fillna('').astype(str).str.replace(' ', '', regex=False).str.replace('　', '', regex=False)

    # 17桁のデータのみをフィルタリング
    isbn_17_digits = cleaned_isbn[cleaned_isbn.str.len() == 17]

    # パターンを分析する関数
    def get_hyphen_pattern(isbn_string):
        parts = isbn_string.split('-')
        lengths = [str(len(p)) for p in parts]
        return "-".join(lengths)

    # 17桁のISBNにパターン分析を適用
    if not isbn_17_digits.empty:
        patterns = isbn_17_digits.apply(get_hyphen_pattern)
        total_pattern_counts.update(patterns)

    print("17桁ISBNのハイフン区切りパターン分析 (from embed_data.parquet):")
    if not total_pattern_counts:
        print("分析対象の17桁ISBNデータが見つかりませんでした。")
    else:
        # 件数が多い順に表示
        for pattern, count in total_pattern_counts.most_common():
            print(f'{pattern}: {count}件')
        
    print("""\n---型式説明---
すべてのデータが 3-1-?-?-1 構成
桁 --- 説明
3 --- 接頭記号、13桁のISBNのデータ. 3,202,743件はすべて 978 となっている. 日本は 978 が割り当てられている
1 --- 国記号、書籍を発行する国や地域を示す記号. 3,202,743件はすべて 4 となっている. 4 は日本出版であることを示す 
? --- 出版者記号、書籍を出版した会社や個人を示す記号. 
? --- 書名記号、各出版者における書籍に割り振られる記号. 重版や増版では同じ記号を割り振るが改訂版などは新たな記号が割り振られる
1 --- チェックデジット、13桁のISBNが正しく表記されているかを確認する記号.
計算方法
  奇数桁 --- 総和 (チェックデジットを除く)
  偶数桁 --- 総和に3をかけたもの
  奇数桁の和と偶数桁の3倍和 = 0 (mod10) になるようにチェックデジットを左辺に加える

例
  978-4-06-276981-5
  奇数桁 --- 9+8+0+2+6+8 = 33
  偶数桁 --- 3(7+4+6+7+9+1) = 102
  33+102+C = 0 (mod10)
  135+C = 0 (mod10)
  C = 5
""")


except ImportError:
    print("エラー: Parquetファイルの読み込みには `pyarrow` ライブラリが必要です。")
    print("pip install pyarrow を実行してください。")
except FileNotFoundError:
    print(f"エラー: {file_path} が見つかりません。")
except Exception as e:
    print(f"エラーが発生しました: {e}")

""" 
---型式説明---
すべてのデータが 3-1-?-?-1 構成
桁 --- 説明
3 --- 接頭記号、13桁のISBNのデータ. 3,202,743件はすべて 978 となっている. 日本は 978 が割り当てられている
1 --- 国記号、書籍を発行する国や地域を示す記号. 3,202,743件はすべて 4 となっている. 4 は日本出版であることを示す 
? --- 出版者記号、書籍を出版した会社や個人を示す記号. 
? --- 書名記号、各出版者における書籍に割り振られる記号. 重版や増版では同じ記号を割り振るが改訂版などは新たな記号が割り振られる
1 --- チェックデジット、13桁のISBNが正しく表記されているかを確認する記号.
計算方法
  奇数桁 --- 総和 (チェックデジットを除く)
  偶数桁 --- 総和に3をかけたもの
  奇数桁の和と偶数桁の3倍和 = 0 (mod10) になるようにチェックデジットを左辺に加える

例
  978-4-06-276981-5
  奇数桁 --- 9+8+0+2+6+8 = 33
  偶数桁 --- 3(7+4+6+7+9+1) = 102
  33+102+C = 0 (mod10)
  135+C = 0 (mod10)
  C = 5
--- 
"""

