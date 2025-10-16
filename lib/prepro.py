import pandas as pd

def clean_time(df):
    df['日付'] = pd.to_datetime(df['日付'])
    df['年'] = df['日付'].dt.year
    df['月'] = df['日付'].dt.month
    df['日'] = df['日付'].dt.day
    df = df.drop('日付', axis=1)
    return df


def delete_space(df, columns):
    for column in columns:
        df[column] = df[column].apply(lambda x: x.replace(' ', '').replace('　', '').strip() if pd.notna(x) else x)
    return df


def normalize_author(df):
    complex_patterns = [
        'さく・え', 'さく／え', '作・絵', '作／絵', '作絵',
        '著・画', '著・監修', '著・訳', '著・作詞', '著・作曲',
        '著・絵', '著／イラスト', '著＋訳', '著作',
        '企画・監修', '企画監修', '編・著', '編著'
    ]
    single_patterns = [
        '監修', '原作', '漫画', 'イラスト',
        '著', '画', '作', '編', '訳', '文', '絵',
        'さく', 'マンガ', 'え', '〔著〕'
    ]

    def process_name(name):
        if pd.isna(name):
            return name
        original = name
        for pattern in complex_patterns:
            if pattern in name:
                name = name.replace(pattern, '')
        for pattern in single_patterns:
            if name.endswith(pattern):
                name = name[:-len(pattern)]
                break
        name = name.replace('他著', '').replace('他', '')
        return name.strip() if name.strip() else original

    df['著者名'] = df['著者名'].apply(process_name)
    return df


def normalize_title(df):
    import re
    def process_title(title):
        if pd.isna(title):
            return title
        original = title
        for delimiter in ['～', '〜', '：', ':']:
            if delimiter in title:
                title = title.split(delimiter)[0]
        if '（' in title:
            if '）' in title:
                title = re.sub(r'（[^）]+）', '', title)
            title = re.sub(r'（.*$', '', title)
        title = re.sub(r'[０-９]+$', '', title)
        title = re.sub(r'[0-9]+$', '', title)
        if title.endswith(('上', '下', '中')) and len(title) > 1:
            title = title[:-1]
        title = re.sub(r'第[一二三四五六七八九十百千]+巻?$', '', title)
        return title.strip() if title.strip() else original

    df['書名'] = df['書名'].apply(process_title)

    return df


def clean_df(df):
    df = df.drop('Unnamed: 0', axis=1)
    df = df.dropna(subset=['出版社', '書名', '著者名', '本体価格'], how='all').copy()
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
    delete_space_columns = [
        'ISBN',
        '書名',
        '出版社',
        '著者名'
    ]
    df = clean_time(df)
    df = delete_space(df, delete_space_columns)
    df = normalize_author(df)
    df = normalize_title(df)

    return df


def count_enc(df, columns):
    for column in columns:
        counts = df[column].value_counts().to_dict()
        df[column] = df[column].map(counts).fillna(0).astype(int)
    return df


def onehot_enc(df, columns):
    for column in columns:
        df = pd.get_dummies(df, columns=[column], dtype=int)
    return df