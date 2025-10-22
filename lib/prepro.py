import pandas as pd
import polars as pl

def clean_time(df):
    df['日付'] = pd.to_datetime(df['日付'])
    #df['年'] = df['日付'].dt.year
    df['月'] = df['日付'].dt.month
    df['日'] = df['日付'].dt.day
    base_date = pd.Timestamp('2024-01-01')
    df['累積日数'] = (df['日付'] - base_date).dt.days+1
    df = df.drop('日付', axis=1)
    time_cols = ['月', '日', '累積日数']
    other_cols = [col for col in df.columns if col not in time_cols]
    df = df[time_cols + other_cols]

    return df


def delete_space(df, columns):
    for col in columns:
        df[col] = df[col].apply(lambda x: x.replace(' ', '').replace('　', '').strip() if pd.notna(x) else x)
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


def fill_missing_class(df):
    isbn_classifications = {
        # 児童書（シールブック系 - まるごとシールブックDXシリーズ）
        "978-4-09-735348-5": {"大分類": "児童　　　　　　　　　　　　　　　　　　", "中分類": "しかけ絵本　　　　　　　　　　　　　　　", "小分類": "シール絵本　　　　　　　　　　　　　　　"},  # シナぷしゅシールブック
        "978-4-09-735603-5": {"大分類": "児童　　　　　　　　　　　　　　　　　　", "中分類": "しかけ絵本　　　　　　　　　　　　　　　", "小分類": "シール絵本　　　　　　　　　　　　　　　"},  # スプラトゥーン3シールブック
        "978-4-09-735605-9": {"大分類": "児童　　　　　　　　　　　　　　　　　　", "中分類": "しかけ絵本　　　　　　　　　　　　　　　", "小分類": "シール絵本　　　　　　　　　　　　　　　"},  # パウ・パトロールシールブック
        "978-4-09-735596-0": {"大分類": "児童　　　　　　　　　　　　　　　　　　", "中分類": "しかけ絵本　　　　　　　　　　　　　　　", "小分類": "シール絵本　　　　　　　　　　　　　　　"},  # ジュラシック・ワールドシールブック
        "978-4-09-735599-1": {"大分類": "児童　　　　　　　　　　　　　　　　　　", "中分類": "しかけ絵本　　　　　　　　　　　　　　　", "小分類": "シール絵本　　　　　　　　　　　　　　　"},  # チキップダンサーズシールブック
        "978-4-09-735604-2": {"大分類": "児童　　　　　　　　　　　　　　　　　　", "中分類": "しかけ絵本　　　　　　　　　　　　　　　", "小分類": "シール絵本　　　　　　　　　　　　　　　"},  # シナぷしゅシールブック
        "978-4-09-735597-7": {"大分類": "児童　　　　　　　　　　　　　　　　　　", "中分類": "しかけ絵本　　　　　　　　　　　　　　　", "小分類": "シール絵本　　　　　　　　　　　　　　　"},  # ミニオンシールブック
        "978-4-09-735606-6": {"大分類": "児童　　　　　　　　　　　　　　　　　　", "中分類": "しかけ絵本　　　　　　　　　　　　　　　", "小分類": "シール絵本　　　　　　　　　　　　　　　"},  # ポケピースシールブック
        "978-4-09-735593-9": {"大分類": "児童　　　　　　　　　　　　　　　　　　", "中分類": "しかけ絵本　　　　　　　　　　　　　　　", "小分類": "シール絵本　　　　　　　　　　　　　　　"},  # うおづらシールブック
        "978-4-09-735608-0": {"大分類": "児童　　　　　　　　　　　　　　　　　　", "中分類": "しかけ絵本　　　　　　　　　　　　　　　", "小分類": "シール絵本　　　　　　　　　　　　　　　"},  # うさまるシールブック
        "978-4-09-735349-2": {"大分類": "児童　　　　　　　　　　　　　　　　　　", "中分類": "創作絵本　　　　　　　　　　　　　　　　", "小分類": "キャラクターその他　　　　　　　　　　　"},  # いないいないばあっ！
        "978-4-09-735611-0": {"大分類": "児童　　　　　　　　　　　　　　　　　　", "中分類": "しかけ絵本　　　　　　　　　　　　　　　", "小分類": "シール絵本　　　　　　　　　　　　　　　"},  # たまごっちシールブック
        "978-4-8133-2675-5": {"大分類": "児童　　　　　　　　　　　　　　　　　　", "中分類": "しかけ絵本　　　　　　　　　　　　　　　", "小分類": "しかけ絵本その他　　　　　　　　　　　　"},  # アナと雪の女王パズルブック

        # コミック
        "978-4-09-735607-3": {"大分類": "コミック　　　　　　　　　　　　　　　　", "中分類": "少年（中高生・一般）　　　　　　　　　　", "小分類": "キャラクターその他　　　　　　　　　　　"},  # 劇場版名探偵コナン
        "978-4-04-067824-5": {"大分類": "コミック　　　　　　　　　　　　　　　　", "中分類": "青年（一般）　　　　　　　　　　　　　　", "小分類": "キャラクターその他　　　　　　　　　　　"},  # 高杉さん家のおべんとう(MFコミックス)
        "978-4-04-067832-0": {"大分類": "コミック　　　　　　　　　　　　　　　　", "中分類": "青年（一般）　　　　　　　　　　　　　　", "小分類": "キャラクターその他　　　　　　　　　　　"},  # 高杉さん家のおべんとう
        "978-4-04-067911-2": {"大分類": "コミック　　　　　　　　　　　　　　　　", "中分類": "青年（一般）　　　　　　　　　　　　　　", "小分類": "キャラクターその他　　　　　　　　　　　"},  # 高杉さん家のおべんとう
        "978-4-04-066101-8": {"大分類": "コミック　　　　　　　　　　　　　　　　", "中分類": "青年（一般）　　　　　　　　　　　　　　", "小分類": "キャラクターその他　　　　　　　　　　　"},  # 高杉さん家のおべんとう
        "978-4-04-066102-5": {"大分類": "コミック　　　　　　　　　　　　　　　　", "中分類": "青年（一般）　　　　　　　　　　　　　　", "小分類": "キャラクターその他　　　　　　　　　　　"},  # 高杉さん家のおべんとう
        "978-4-04-066900-7": {"大分類": "コミック　　　　　　　　　　　　　　　　", "中分類": "青年（一般）　　　　　　　　　　　　　　", "小分類": "キャラクターその他　　　　　　　　　　　"},  # 高杉さん家のおべんとう
        "978-4-02-275066-2": {"大分類": "コミック　　　　　　　　　　　　　　　　", "中分類": "児童　　　　　　　　　　　　　　　　　　", "小分類": "キャラクターその他　　　　　　　　　　　"},  # 落第忍者乱太郎(ギャグ漫画・児童向け)

        # 高校学参(大学受験参考書)
        "978-4-407-36376-0": {"大分類": "高校学参　　　　　　　　　　　　　　　　", "中分類": "全般　　　　　　　　　　　　　　　　　　", "小分類": "参考書　　　　　　　　　　　　　　　　　"},  # 大学入試短期集中ゼミ数学I+A
        "978-4-407-36382-1": {"大分類": "高校学参　　　　　　　　　　　　　　　　", "中分類": "全般　　　　　　　　　　　　　　　　　　", "小分類": "参考書　　　　　　　　　　　　　　　　　"},  # 大学入試短期集中ゼミ
        "978-4-407-36372-2": {"大分類": "高校学参　　　　　　　　　　　　　　　　", "中分類": "全般　　　　　　　　　　　　　　　　　　", "小分類": "参考書　　　　　　　　　　　　　　　　　"},  # 大学入学共通テスト数学

        # 就職・資格
        "978-4-407-36383-8": {"大分類": "就職・資格　　　　　　　　　　　　　　　", "中分類": "全般　　　　　　　　　　　　　　　　　　", "小分類": "参考書　　　　　　　　　　　　　　　　　"},  # 看護・医療系のための数学
        "978-4-86639-765-8": {"大分類": "語学　　　　　　　　　　　　　　　　　　", "中分類": "全般　　　　　　　　　　　　　　　　　　", "小分類": "参考書　　　　　　　　　　　　　　　　　"},  # TOEFL iBTテスト総合対策

        # 月刊誌・隔月刊誌
        "978-4-02-275476-9": {"大分類": "月刊誌　　　　　　　　　　　　　　　　　", "中分類": "全般　　　　　　　　　　　　　　　　　　", "小分類": "キャラクターその他　　　　　　　　　　　"},  # HONKOWA(ホラー・オカルト隔月刊誌)
        "978-4-8334-4135-3": {"大分類": "月刊誌　　　　　　　　　　　　　　　　　", "中分類": "全般　　　　　　　　　　　　　　　　　　", "小分類": "キャラクターその他　　　　　　　　　　　"},  # VOGUE JAPAN 25周年スペシャルエディション
    }

    # 分類情報を付与
    for isbn, classification in isbn_classifications.items():
        mask = (df['ISBN'] == isbn) & (df['大分類'].isna())
        if mask.any():
            df.loc[mask, '大分類'] = classification['大分類']
            df.loc[mask, '中分類'] = classification['中分類']
            df.loc[mask, '小分類'] = classification['小分類']

    return df


def fill_publisher_by_ISBN(df, columns):
    for prefix, publisher in columns.items():
        df.loc[df['ISBN'].astype(str).str.startswith(prefix), '出版社'] = publisher
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
    delete_space_columns = [
        'ISBN',
        '書名',
        '出版社',
        '著者名'
    ]
    enc_columns = [
        '出版社',
        '書名',
        '著者名'
    ]
    df = clean_time(df)
    df = fill_publisher_by_ISBN(df, isbn_to_publisher)
    df = delete_space(df, delete_space_columns)
    df = normalize_author(df)
    df = normalize_title(df)
    df = fill_missing_class(df)
    df = enc(df, enc_columns)

    return df


def count_enc(df, columns):
    for col in columns:
        counts = df[col].value_counts().to_dict()
        df[col] = df[col].map(counts).fillna(0).astype(int)
    return df


def onehot_enc(df, columns):
    for col in columns:
        df = pd.get_dummies(df, columns=[col], dtype=int)
    return df


def enc(df, columns):
    """
    出現頻度順にエンコード（pandas版）
    - 最頻値: ユニーク数(最大値)
    - 最低頻度: 1(最小値)
    - 同頻度の場合: 値の昇順で処理
    - 元のカラムの「右隣」に <col>_enc を追加
    """

    for col in columns:
        vc = df[col].value_counts().reset_index()
        vc.columns = [col, "count"]
        vc = vc.sort_values(["count", col], ascending=[False, True])
        vc["encoding"] = range(len(vc), 0, -1)
        enc_map = dict(zip(vc[col], vc["encoding"]))
        enc_values = df[col].map(enc_map)
        insert_pos = df.columns.get_loc(col) + 1
        df.insert(insert_pos, f"{col}_enc", enc_values)

    return df

