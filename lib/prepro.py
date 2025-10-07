import pandas as pd
import plotly.express as px

def delete_space(df, column):
    df[column] = df[column].apply(lambda x: x.replace(' ', '').replace('　', '').strip() if pd.notna(x) else x)
    return df

def normalize_author(df, column):
    def _normalize(name):
        if pd.isna(name):
            return name
        original = name
        complex_patterns = [
            'さく・え', 'さく／え', '作・絵', '作／絵', '作絵',
            '著・画', '著・監修', '著・訳', '著・作詞', '著・作曲',
            '著・絵', '著／イラスト', '著＋訳', '著作',
            '企画・監修', '企画監修', '編・著', '編著'
        ]
        for pattern in complex_patterns:
            if pattern in name:
                name = name.replace(pattern, '')
        single_patterns = [
            '監修', '原作', '漫画', 'イラスト',
            '著', '画', '作', '編', '訳', '文', '絵',
            'さく', 'マンガ', 'え', '〔著〕'
        ]
        for pattern in single_patterns:
            if name.endswith(pattern):
                name = name[:-len(pattern)]
                break  # 一度だけ削除
        name = name.replace('他著', '').replace('他', '')
        return name.strip() if name.strip() else original

    df[column] = df[column].apply(_normalize)
    return df

def normalize_title(df, column):
    import re
    def _normalize(title):
        if pd.isna(title):
            return title
        original = title
        for delimiter in ['～', '〜', '：', ':']:
            if delimiter in title:
                title = title.split(delimiter)[0]
        # 括弧とその中身を削除（閉じ括弧がない場合も対応）
        if '（' in title:
            if '）' in title:
                # 通常の括弧ペアを削除
                title = re.sub(r'（[^）]+）', '', title)
            # 開き括弧以降を削除（閉じ括弧がない場合）
            title = re.sub(r'（.*$', '', title)
        title = re.sub(r'[０-９]+$', '', title)
        title = re.sub(r'[0-9]+$', '', title)
        if title.endswith('上') or title.endswith('下') or title.endswith('中'):
            if len(title) > 1:
                title = title[:-1]
        title = re.sub(r'第[一二三四五六七八九十百千]+巻?$', '', title)
        return title.strip() if title.strip() else original

    df[column] = df[column].apply(_normalize)
    return df

def clean_df(df):
    # 著者名・出版社・書名・本体価格が全て欠損している行を削除
    df = df.dropna(subset=['著者名', '出版社', '書名', '本体価格'], how='all').copy()

    # 17件の出版社欠損値を補完 + 出版社の表記ゆれを補完
    isbn_to_publisher = {
        "978-4-939094-": "福島テレビ",
        "978-4-341-": "ごま書房新社",
        "978-4-387-": "サンリオ",
        "978-4-480-": "筑摩書房",
        "978-4-7698-": "潮書房光人新社",
        "978-4-7770-": "ネコ・パブリッシング",
        "978-4-7796-": "三栄",
        "978-4-7999-": "文溪堂",
        "978-4-8069-": "つちや書店",
        "978-4-88144-": "創藝社",
        "978-4-89423-": "文溪堂",
        }
    for prefix, publisher in isbn_to_publisher.items():
        df.loc[df['ISBN'].astype(str).str.startswith(prefix), '出版社'] = publisher
    # 出版社名の空白削除
    df = delete_space(df, '出版社')
    #日付
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

    #on-hot encoding
    df['出版社'] = df['出版社'].map(publisher_counts).fillna(0).astype(int)
    store_dummies = pd.get_dummies(df['書店コード'], prefix='書店').astype(int)
    df = pd.concat([df, store_dummies], axis=1)
    df = pd.get_dummies(df, columns=['大分類'], dtype=int)

    # 各カテゴリの出現回数を計算
    count_map_1 = df['中分類'].value_counts()
    count_map_2 = df['小分類'].value_counts()
    # 既存の列を上書きしてカウントエンコーディングを適用
    df['中分類'] = df['中分類'].map(count_map_1).fillna(0).astype(int)
    df['小分類'] = df['小分類'].map(count_map_2).fillna(0).astype(int)

    #drop columns

    drop_columns=[
        '日付',
        '書店コード'
    ]
    for column in drop_columns:
        df = df.drop(columns=[column])

    return df