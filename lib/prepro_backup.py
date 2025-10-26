import pandas as pd

def clean_time(df):
    df['日付'] = pd.to_datetime(df['日付'])
    #df['年'] = df['日付'].dt.year
    df['月'] = df['日付'].dt.month
    df['日'] = df['日付'].dt.day
    base_date = pd.Timestamp('2024-01-01')
    df['累積日数'] = (df['日付'] - base_date).dt.days+1
    time_cols = ['月', '日', '累積日数']
    other_cols = [col for col in df.columns if col not in time_cols]
    df = df[time_cols + other_cols]
    return df


def delete_space(df, columns):
    for col in columns:
        df[col] = df[col].apply(lambda x: x.replace(' ', '').replace('　', '').strip() if pd.notna(x) else x)
    return df


def normalize_author(df):
    import re
    patterns = [
        '著', '監修', '画', '作', '編', '漫画', '他著', '編著', '原作', '文', '他',
        'さく', '絵', '作・絵', '他編', '作画', '撮影', 'ぶん', '訳', 'え',
        '他編著', '他監修', '編集', '写真', '解説', '監', '調査執筆', 'マンガ',
        '原案', '写真・文', '講師', '総監修', '文・写真', 'イラスト', '文・絵',
        '編訳', '監修・著', 'まんが', '絵と文', '料理', '訳注', 'さく・え',
        '編・著', '原著', '校注', '他作', '小説', '他画', '脚本', '責任編集',
        '編集主幹', '詩', '執筆', '病態監修', '原作・絵', '他訳', '医学監修',
        '責任監修', '協力', '作絵', '書', '監訳', '他絵', '編集代表', '著・監修',
        '語り', '詞', '構成', '他文', 'ぶん・え', '絵・文', '作・画', '訳・解説',
        '著作', '述', '他編集', '構成・絵', '出題', '編曲', '作／絵', 'ストーリー',
        '翻訳', '著・演奏', '著・絵', '写真と文', '全訳注', '他監', '指導',
        '文と絵', '聞き手', '他校注', '企画', '選', '著・写真', '作家',
        '英文解説', 'マンガ・文', '考案', 'レシピ', '特別講師', '原作／絵',
        '制作', 'ことば', '監修・写', '著・画', '構成・文', '他マンガ',
        '栄養監修', '訳編', '監修・文', '特別編集', '訳・監修', '編纂',
        '医療解説', 'デザイン', '写真・著', '再話', '俳句', '塗り絵', '他脚本',
        '他著作', '脚本・絵'
    ]
    """出現回数1回
    '著・訳', '著／イラスト', '著＋訳', '企画・監修',
    '企画監修', 'さく／え', '漢字監修', '植物監修', '編集・執筆',
    '地図監修', '英語監修', '日本語監修', '医師監修', '著・イラスト',
    '記事監修', '解説・監修', '監修執筆', '文・画', '図案', '手本・監修',
    '監修・考案', '俳句監修', '監修代表', '代表監修', '編集・文', '絵・著',
    '監修・協力', 'イラスト・文', '他全訳注', '他選・文', '漫画・文',
    '編修主幹', '写真・監修', '監修・執筆', '執筆・解説', '監修・解説',
    '編修代表', '編集協力', '取材・文', '撮影・監修', '監修・制作',
    '編集・構成', '編集責任', '監修・料理', '食事指導', '監修・作画',
    '絵・監訳', '作詞者', '家紋監修', 'しかけ', 'にんぎょう・え', '編修',
    '他料理', '他写真', '調理', 'パズル制作', '編訳・解説', '編・訳',
    '文・漫画', 'ぶ', '語り手', '他編集委員', '他編集代表',
    '手本', '著・作詞', '著・作曲', '他選', '編・写真', 'ノベライズ',
    '文・構成', '他講師', '編著代表', '他調査執筆', '編者代表', '主幹',
    '原訳', '校訂・訳', '主編', '編著・監修', '編・解説', '御作曲',
    '序', '原詩', '画と文', '原案＆作画', '原案・文', '他校訂・訳',
    'え・ぶん', '原案・解説', '監・著', '監修協力', '影絵', '校訂', '案',
    '他え', '訳・著', '技術指導', '朗読', '解説・訳', '注訳', 'しゃしん',
    '詩・文', '短歌・文', '講義', '詩人', '訳・注', '現代語訳', '著・装画',
    '監修訳注', '他執筆', '板書', '本文', '案と絵', '絵と案', '文章',
    '他原作', 'コミック', '選・著', '原作・原案', '原案・脚本', '監督',
    '推薦', '文と写真', '原作・脚本', '編／写真', '芝居', 'ほか著',
    '企画・編集', '改訂監修', '再話・絵', '指導・編曲', '他責任編集',
    '写真協力', '編注', '原著作', '主幹編著', '〔著〕'
    """
    patterns_sorted = sorted(patterns, key=len, reverse=True)
    combined_pattern = '|'.join(re.escape(p) for p in patterns_sorted)
    regex = re.compile(f'　(?:{combined_pattern})$')
    df['著者名'] = (df['著者名'].str.replace(regex, '', regex=True).str.replace('　', '', regex=False))
    return df


def normalize_title(df):
    import re
    specific_patterns = {
        '変な家　文庫版': '変な家',
        '呪術廻戦　　　０　東京都立呪術高等専門学': '呪術廻戦_0',
        '成瀬は天下を取りにいく_1': '成瀬は天下を取りにいく',
        '傲慢と善良_1': '傲慢と善良',
        '白鳥とコウモリ': '白鳥とコウモリ_上',
        '薬屋のひとりごと': '薬屋のひとりごと_1'
    }
    # 薬屋のひとりごとシリーズのパターン（～以降のサブタイトルを除去）
    pattern_kusuriya = r'^薬屋のひとりごと[～〜].+?_?(\d+)$'
    pattern_with_suffix = r'^(.+?)　+([０-９]+)　+(.+)$'
    pattern_volume = r'^(.+?)　+([０-９]+)$'
    pattern_text_suffix = r'^(.+?)　+(上|下|前編|後編|第[一二三四五六七八九十]+巻)$'
    trans_table = str.maketrans('０１２３４５６７８９', '0123456789')
    def process_title(title):
        if pd.isna(title):
            return title
        match_kusuriya = re.match(pattern_kusuriya, title)
        if match_kusuriya:
            volume = match_kusuriya.group(1)
            return f"薬屋のひとりごと_{volume}"
        match = re.match(pattern_with_suffix, title)
        if match:
            base_title = match.group(1)
            volume = match.group(2).translate(trans_table)
            suffix = match.group(3)
            return f"{base_title}_{suffix}{volume}"
        match = re.match(pattern_volume, title)
        if match:
            base_title = match.group(1)
            volume = match.group(2).translate(trans_table)
            return f"{base_title}_{volume}"
        match = re.match(pattern_text_suffix, title)
        if match:
            base_title = match.group(1)
            suffix = match.group(2)
            return f"{base_title}_{suffix}"
        if title in specific_patterns:
            return specific_patterns[title]
        return title
    df['書名'] = df['書名'].apply(process_title)
    return df


def remove_volume_number(df):
    """
    書名から「_巻数」部分を除去する
    前提: normalize_title()で「書名_巻数」形式に統一されていること
    例: 「タイトル_1」→「タイトル」
        「タイトル_上」→「タイトル」
        「タイトル」→「タイトル」（巻数なしはそのまま）
    """
    def process_volume(title):
        if pd.isna(title):
            return title
        # _以降を除去（巻数部分）
        if '_' in title:
            return title.split('_')[0]
        return title

    df['書名'] = df['書名'].apply(process_volume)
    return df


def fill_missing_class(df):
    isbn_classifications = {
        "978-4-09-735348-5": {"大分類": "児童", "中分類": "しかけ絵本", "小分類": "シール絵本"},  # シナぷしゅシールブック
        "978-4-09-735603-5": {"大分類": "児童", "中分類": "しかけ絵本", "小分類": "シール絵本"},  # スプラトゥーン3シールブック
        "978-4-09-735605-9": {"大分類": "児童", "中分類": "しかけ絵本", "小分類": "シール絵本"},  # パウ・パトロールシールブック
        "978-4-09-735596-0": {"大分類": "児童", "中分類": "しかけ絵本", "小分類": "シール絵本"},  # ジュラシック・ワールドシールブック
        "978-4-09-735599-1": {"大分類": "児童", "中分類": "しかけ絵本", "小分類": "シール絵本"},  # チキップダンサーズシールブック
        "978-4-09-735604-2": {"大分類": "児童", "中分類": "しかけ絵本", "小分類": "シール絵本"},  # シナぷしゅシールブック
        "978-4-09-735597-7": {"大分類": "児童", "中分類": "しかけ絵本", "小分類": "シール絵本"},  # ミニオンシールブック
        "978-4-09-735606-6": {"大分類": "児童", "中分類": "しかけ絵本", "小分類": "シール絵本"},  # ポケピースシールブック
        "978-4-09-735593-9": {"大分類": "児童", "中分類": "しかけ絵本", "小分類": "シール絵本"},  # うおづらシールブック
        "978-4-09-735608-0": {"大分類": "児童", "中分類": "しかけ絵本", "小分類": "シール絵本"},  # うさまるシールブック
        "978-4-09-735349-2": {"大分類": "児童", "中分類": "創作絵本", "小分類": "キャラクターその他"},  # いないいないばあっ！
        "978-4-09-735611-0": {"大分類": "児童", "中分類": "しかけ絵本", "小分類": "シール絵本"},  # たまごっちシールブック
        "978-4-8133-2675-5": {"大分類": "児童", "中分類": "しかけ絵本", "小分類": "しかけ絵本その他"},  # アナと雪の女王パズルブック

        "978-4-09-735607-3": {"大分類": "コミック", "中分類": "少年（中高生・一般）", "小分類": "キャラクターその他"},  # 劇場版名探偵コナン
        "978-4-04-067824-5": {"大分類": "コミック", "中分類": "青年（一般）", "小分類": "キャラクターその他"},  # 高杉さん家のおべんとう(MFコミックス)
        "978-4-04-067832-0": {"大分類": "コミック", "中分類": "青年（一般）", "小分類": "キャラクターその他"},  # 高杉さん家のおべんとう
        "978-4-04-067911-2": {"大分類": "コミック", "中分類": "青年（一般）", "小分類": "キャラクターその他"},  # 高杉さん家のおべんとう
        "978-4-04-066101-8": {"大分類": "コミック", "中分類": "青年（一般）", "小分類": "キャラクターその他"},  # 高杉さん家のおべんとう
        "978-4-04-066102-5": {"大分類": "コミック", "中分類": "青年（一般）", "小分類": "キャラクターその他"},  # 高杉さん家のおべんとう
        "978-4-04-066900-7": {"大分類": "コミック", "中分類": "青年（一般）", "小分類": "キャラクターその他"},  # 高杉さん家のおべんとう
        "978-4-02-275066-2": {"大分類": "コミック", "中分類": "児童", "小分類": "キャラクターその他"},  # 落第忍者乱太郎(ギャグ漫画・児童向け)

        "978-4-407-36376-0": {"大分類": "高校学参", "中分類": "全般", "小分類": "参考書"},  # 大学入試短期集中ゼミ数学I+A
        "978-4-407-36382-1": {"大分類": "高校学参", "中分類": "全般", "小分類": "参考書"},  # 大学入試短期集中ゼミ
        "978-4-407-36372-2": {"大分類": "高校学参", "中分類": "全般", "小分類": "参考書"},  # 大学入学共通テスト数学

        "978-4-407-36383-8": {"大分類": "就職・資格", "中分類": "全般", "小分類": "参考書"},  # 看護・医療系のための数学
        "978-4-86639-765-8": {"大分類": "語学", "中分類": "全般", "小分類": "参考書"},  # TOEFL iBTテスト総合対策

        "978-4-02-275476-9": {"大分類": "月刊誌", "中分類": "全般", "小分類": "キャラクターその他"},  # HONKOWA(ホラー・オカルト隔月刊誌)
        "978-4-8334-4135-3": {"大分類": "月刊誌", "中分類": "全般", "小分類": "キャラクターその他"},  # VOGUE JAPAN 25周年スペシャルエディション
    }

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
    df = clean_time(df)
    df = fill_publisher_by_ISBN(df, isbn_to_publisher)
    df = delete_space(df, delete_space_columns)
    df = normalize_author(df)
    df = normalize_title(df)
    #df = remove_volume_number(df)
    df = fill_missing_class(df)
    """
    enc_columns = [
        '出版社',
        '書名',
        '著者名'
    ]
    df = enc(df, enc_columns)
    """
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
    出現頻度順にエンコード(pandas版)
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


def label_enc(df, columns):
    from sklearn.preprocessing import LabelEncoder
    for col in columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].fillna('Unknown').astype(str))

    return df

