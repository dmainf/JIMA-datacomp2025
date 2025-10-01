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

    drop_columns=[
        '日付',
    ]
    for column in drop_columns:
        df = df.drop(columns=[column])

    return df