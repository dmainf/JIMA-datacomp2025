# データ分析プロジェクト

## ディレクトリ構成
.
├── README.md：本ファイル
├── a.py：データ読み込みとクリーニング後の統計情報を出力
├── embeding_data.py：書名と著者名をSentenceTransformerでベクトル化し、parquet形式で保存
├── lib/
│   ├── prepro.py：データ前処理関数（日付変換、スペース削除、著者名・書名の正規化、エンコーディング等）
│   ├── fueture_eng.py：テキストデータをSentenceTransformerでベクトル化し、PCAで次元削減
│   └── model_cache.py：SentenceTransformerモデルをキャッシュして再利用（M4 Mac MPS対応）
└── dist/
    ├── count_author_patterns.py：著者名に含まれる「著」「監修」などのパターンの出現回数を集計
    ├── extract_pattern2.py：特定パターンのデータを抽出
    ├── plot_dist.py：指定カラムの分布（上位20件）と出現回数の頻度分布をプロット
    ├── plot_date.py：日付の時系列分布をプロット
    ├── plot_pos_sales.py：POS販売冊数の分布をプロット
    ├── plot_overlay_comparison.py：複数の分布を重ねて比較プロット
    ├── plot_price.py：本体価格の分布をプロット
    ├── how_to_use_dist.py：plot_dist.pyの使用例
    ├── SunburstChart.py：階層データをサンバーストチャートで可視化
    ├── data/
    │   ├── pattern2_no_author.csv：著者名除外パターン2のデータ（422MB、Git LFS管理）
    │   ├── 書名_一覧.csv：書名の一覧（8MB、Git LFS管理）
    │   ├── 書名_正規化後_一覧.csv：正規化後の書名一覧（5.7MB、Git LFS管理）
    │   ├── 著者名_一覧.csv：著者名の一覧（1.3MB、Git LFS管理）
    │   └── 著者名_正規化後_一覧.csv：正規化後の著者名一覧（876KB、Git LFS管理）
    └── figure/
        ├── 書名_distribution.png：書名の分布グラフ（上位20件）
        ├── 書名_counts_distribution.png：書名の出現回数頻度分布
        ├── 書名_counts_dist_loglog.png：書名の出現回数頻度分布（両対数）
        ├── 著者名_distribution.png：著者名の分布グラフ（上位20件）
        ├── 著者名_counts_distribution.png：著者名の出現回数頻度分布
        ├── 著者名_counts_dist_loglog.png：著者名の出現回数頻度分布（両対数）
        ├── 著者名_書名_counts_comparison_loglog.png：著者名と書名の出現回数比較（両対数）
        ├── 出版社_dist.png：出版社の分布
        ├── 出版社_distribution.png：出版社の分布グラフ（上位20件）
        ├── 出版社_counts_dist_loglog.png：出版社の出現回数頻度分布（両対数）
        ├── 大分類_dist.png：大分類の分布
        ├── 大分類_counts_dist_loglog.png：大分類の出現回数頻度分布（両対数）
        ├── 中分類_dist.png：中分類の分布
        ├── 中分類_counts_dist_loglog.png：中分類の出現回数頻度分布（両対数）
        ├── 小分類_dist.png：小分類の分布
        ├── 小分類_counts_dist_loglog.png：小分類の出現回数頻度分布（両対数）
        ├── 日付_distribution.png：日付の時系列分布
        ├── 本体価格_dist.png：本体価格の分布
        ├── POS販売冊数_dist.png：POS販売冊数の分布
        ├── december_大分類_dist.png：12月の大分類分布
        ├── パターン2_大分類_dist.png：パターン2の大分類分布
        └── trush/：未使用・テスト用の図