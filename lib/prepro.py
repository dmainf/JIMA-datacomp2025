import pandas as pd

def clean_df(df):
    df['日付'] = pd.to_datetime(df['日付'])
    df['year'] = df['日付'].dt.year
    df['month'] = df['日付'].dt.month
    df['day'] = df['日付'].dt.day
    book_counts = df['書名'].value_counts().to_dict()
    df['書名'] = df['書名'].map(book_counts).fillna(0).astype(int)
    author_counts = df['著者名'].value_counts().to_dict()
    df['著者名'] = df['著者名'].map(author_counts).fillna(0).astype(int)
    df['本体価格'] = df['本体価格'].fillna(0)
    store_dummies = pd.get_dummies(df['書店コード'], prefix='store').astype(int)
    df = pd.concat([df, store_dummies], axis=1)
    isbn_counts = df['ISBN'].value_counts().to_dict()
    df['ISBN'] = df['ISBN'].map(isbn_counts).fillna(0).astype(int)
    publisher_counts = df['出版社'].value_counts().to_dict()
    df['出版社'] = df['出版社'].map(publisher_counts).fillna(0).astype(int)

    drop_columns=[
        '日付',
        '書店コード'
    ]
    for column in drop_columns:
        df = df.drop(columns=[column])

    return df