import pandas as pd

def clean_df(df):
    # 著者名・出版社・書名・本体価格が全て欠損している行を削除
    df = df.dropna(subset=['著者名', '出版社', '書名', '本体価格'], how='all')
    #
    df['日付'] = pd.to_datetime(df['日付'])
    df['年'] = df['日付'].dt.year
    df['月'] = df['日付'].dt.month
    df['日'] = df['日付'].dt.day
    #count encoding
    book_counts = df['書名'].value_counts().to_dict()
    df['書名'] = df['書名'].map(book_counts).fillna(0).astype(int)
    author_counts = df['著者名'].value_counts().to_dict()
    df['著者名'] = df['著者名'].map(author_counts).fillna(0).astype(int)
        #df['本体価格'] = df['本体価格'].fillna(0) 書名の欠損値の数と全く同じだったから行ごと削除で全部落とせてる
    isbn_counts = df['ISBN'].value_counts().to_dict()
    df['ISBN'] = df['ISBN'].map(isbn_counts).fillna(0).astype(int)
    publisher_counts = df['出版社'].value_counts().to_dict()
    #on-hot encoding because of small num of columns
    df['出版社'] = df['出版社'].map(publisher_counts).fillna(0).astype(int)
    store_dummies = pd.get_dummies(df['書店コード'], prefix='書店').astype(int)
    df = pd.concat([df, store_dummies], axis=1)
    # ワンホットエンコーディングを実行
    df = pd.get_dummies(df, columns=['大分類'], dtype=int)
    # 各カテゴリの出現回数を計算
    count_map_1 = df['中分類'].value_counts()
    count_map_2 = df['小分類'].value_counts()
    # 既存の列を上書きしてカウントエンコーディングを適用
    df['中分類'] = df['中分類'].map(count_map_1)
    df['小分類'] = df['小分類'].map(count_map_2)

    drop_columns=[
        '日付',
        '書店コード'
    ]
    for column in drop_columns:
        df = df.drop(columns=[column])

    return df