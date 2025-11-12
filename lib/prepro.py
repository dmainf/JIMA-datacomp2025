import pandas as pd

def clean_time(df):
    df['日付'] = pd.to_datetime(df['日付']).dt.normalize()
    df['月'] = df['日付'].dt.month
    df['日'] = df['日付'].dt.day
    base_date = pd.Timestamp('2024-01-01')
    df['累積日数'] = (df['日付'] - base_date).dt.days
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
    trans_table = str.maketrans('０１２３４５６７８９', '0123456789')
    def process_title(title):
        if pd.isna(title):
            return title
        original_title = title
        title = re.sub(r'　+(上)$', '　1', title)
        title = re.sub(r'　+(下)$', '　2', title)
        title = re.sub(r'　+(前編)$', '　1', title)
        title = re.sub(r'　+(後編)$', '　2', title)
        # 1. 葬送のフリーレン
        if '葬送のフリーレン' in title:
            # 小説
            if '小説' in title or '前奏' in title or '魂の眠る地' in title:
                return '葬送のフリーレン_小説_0'
            # 関連書籍
            if any(x in title for x in ['公式ファンブック', '画集', 'アンソロジー', 'クリアファイル',
                                        'ポストカード', 'ＴＶアニメ', 'コミック付箋', 'ポスターコレクション',
                                        'てれびくん', 'カレンダ']):
                return '葬送のフリーレン_関連書籍_0'
            # 特装版を除去
            title = re.sub(r'　+(特装版|限定版)$', '', title)
            # 本編巻数
            match = re.search(r'葬送のフリーレン　+([０-９\d]+)', title)
            if match:
                vol = match.group(1).translate(trans_table)
                return f'葬送のフリーレン_original_{vol}'
            return '葬送のフリーレン_original_0'
        # 2. 怪獣８号
        if '怪獣８号' in title:
            # スピンオフ: side B
            if 'ｓｉｄｅ　Ｂ' in title:
                match = re.search(r'[　\s]+([０-９\d]+)', title)
                if match:
                    vol = match.group(1).translate(trans_table)
                    return f'怪獣８号_sideB_{vol}'
                return '怪獣８号_sideB_0'
            # スピンオフ: RELAX
            if 'ＲＥＬＡＸ' in title:
                match = re.search(r'[　\s]+([０-９\d]+)', title)
                if match:
                    vol = match.group(1).translate(trans_table)
                    return f'怪獣８号_RELAX_{vol}'
                return '怪獣８号_RELAX_0'
            # 関連書籍
            if any(x in title for x in ['密着', '防衛隊', 'フォルティチュード']):
                return '怪獣８号_関連書籍_0'
            # 本編
            match = re.search(r'怪獣８号　+([０-９\d]+)', title)
            if match:
                vol = match.group(1).translate(trans_table)
                return f'怪獣８号_original_{vol}'
            return '怪獣８号_original_0'
        # 3. 呪術廻戦
        if '呪術廻戦' in title:
            # 小説
            if '劇場版' in title and ('ノベライズ' in title or '０' in title):
                return '呪術廻戦_劇場版0ノベライズ_0'
            if '夜明けのいばら道' in title or '逝く夏と還る秋' in title:
                return '呪術廻戦_小説_0'
            # 関連書籍
            if any(x in title for x in ['公式ファンブック', 'ＴＶアニメ', 'で英語', '塗絵帳', 'カレンダー', '楽しむ']):
                return '呪術廻戦_関連書籍_0'
            # 特装版・同梱版を除去
            title = re.sub(r'　+(同梱版|カレンダー同梱版)$', '', title)
            # 0巻
            if '　　　０　' in title or '　　０　' in title:
                return '呪術廻戦_original_0'
            # 本編
            match = re.search(r'呪術廻戦　+([０-９\d]+)', title)
            if match:
                vol = match.group(1).translate(trans_table)
                return f'呪術廻戦_original_{vol}'
            return '呪術廻戦_original_0'
        # 4. ＯＮＥ　ＰＩＥＣＥ
        if 'ＯＮＥ　ＰＩＥＣＥ' in title:
            # キャラクタースピンオフ
            characters = ['ロロノア・ゾロ', 'サンジ', 'ナミ', 'ウソップ', 'ニコ・ロビン',
                         'トニートニー・チョッ', 'ブルック', 'フランキー']
            for char in characters:
                if char in title:
                    return 'ＯＮＥ　ＰＩＥＣＥ_スピンオフ_0'
            # 映画関連
            if 'ＦＩＬＭ' in title or '劇場版' in title:
                return 'ＯＮＥ　ＰＩＥＣＥ_映画関連_0'
            # ノベライズ
            if 'ｎｏｖｅｌ' in title:
                return 'ＯＮＥ　ＰＩＥＣＥ_novel_0'
            # 学園シリーズ
            if '学園' in title:
                match = re.search(r'[０-９\d]+', title)
                if match:
                    vol = match.group(0).translate(trans_table)
                    return f'ＯＮＥ　ＰＩＥＣＥ_学園_{vol}'
                return 'ＯＮＥ　ＰＩＥＣＥ_学園_0'
            # エピソードA
            if 'ｅｐｉｓｏｄｅＡ' in title:
                match = re.search(r'[０-９\d]+', title)
                if match:
                    vol = match.group(0).translate(trans_table)
                    return f'ＯＮＥ　ＰＩＥＣＥ_episodeA_{vol}'
                return 'ＯＮＥ　ＰＩＥＣＥ_episodeA_0'
            # magazine
            if 'ｍａｇａｚｉｎ' in title:
                match = re.search(r'[０-９\d]+', title)
                if match:
                    vol = match.group(0).translate(trans_table)
                    return f'ＯＮＥ　ＰＩＥＣＥ_magazine_{vol}'
                return 'ＯＮＥ　ＰＩＥＣＥ_magazine_0'
            # ヒロインズ、カードゲーム、ビブルカード、カレンダーなど
            if any(x in title for x in ['ヒロインズ', 'ＣＡＲＤ　ＧＡＭＥ', 'ビブルカード',
                                       'カレンダー', 'るるぶ', 'ＢＬＵＥ', 'ＲＡＩＮＢＯＷ',
                                       'ＷＨＩＴＥ', 'ＲＥＤ　ＧＲＡＮＤ', 'ＱＵＩＺ']):
                return 'ＯＮＥ　ＰＩＥＣＥ_関連書籍_0'
            # BOXシリーズ
            if 'ＢＯＸ' in title:
                return 'ＯＮＥ　ＰＩＥＣＥ_BOX_0'
            # 本編
            match = re.search(r'ＯＮＥ　ＰＩＥＣＥ　+([０-９\d]+)$', title)
            if match:
                vol = match.group(1).translate(trans_table)
                return f'ＯＮＥ　ＰＩＥＣＥ_original_{vol}'
            return 'ＯＮＥ　ＰＩＥＣＥ_original_0'
        # 5. 変な家
        if title.startswith('変な家'):
            if '文庫版' in title:
                return '変な家_文庫版_0'
            match = re.search(r'変な家　+([０-９\d]+)', title)
            if match:
                vol = match.group(1).translate(trans_table)
                return f'変な家_original_{vol}'
            return '変な家_original_0'
        # 6. ブルーロック
        if 'ブルーロック' in title:
            # スピンオフ: EPISODE凪
            if 'ＥＰＩＳＯＤＥ' in title:
                match = re.search(r'[　\s]+([０-９\d]+)', title)
                if match:
                    vol = match.group(1).translate(trans_table)
                    return f'ブルーロック_EPISODE凪_{vol}'
                return 'ブルーロック_EPISODE凪_0'
            # 小説: EPISODE凪
            if '小説ブルーロック－ＥＰＩＳＯＤＥ凪' in title or '小説　ブルーロック－ＥＰＩＳＯＤＥ凪' in title:
                match = re.search(r'[　\s]+([０-９\d]+)', title)
                if match:
                    vol = match.group(1).translate(trans_table)
                    return f'ブルーロック_小説EPISODE凪_{vol}'
                return 'ブルーロック_小説EPISODE凪_0'
            # 小説
            if title.startswith('小説'):
                match = re.search(r'[　\s]+([０-９\d]+)', title)
                if match:
                    vol = match.group(1).translate(trans_table)
                    return f'ブルーロック_小説_{vol}'
                return 'ブルーロック_小説_0'
            # 関連書籍
            if any(x in title for x in ['キャラクターブック', 'ＴＶアニメ', '漢字ドリル', 'コンプリート',
                                        'ポストカード', 'クリアファイル', 'トランプ', 'ぬりえ', '都道府県']):
                return 'ブルーロック_関連書籍_0'
            # 特装版を除去
            title = re.sub(r'　+特装版$', '', title)
            # 本編
            match = re.search(r'ブルーロック　+([０-９\d]+)', title)
            if match:
                vol = match.group(1).translate(trans_table)
                return f'ブルーロック_original_{vol}'
            return 'ブルーロック_original_0'
        # 7. 僕のヒーローアカデミア
        if '僕のヒーローアカデミア' in title:
            # スピンオフ: ヴィジランテ
            if 'ヴィジランテ' in title:
                match = re.search(r'[　\s]+([０-９\d]+)', title)
                if match:
                    vol = match.group(1).translate(trans_table)
                    return f'僕のヒーローアカデミア_ヴィジランテ_{vol}'
                return '僕のヒーローアカデミア_ヴィジランテ_0'
            # スピンオフ: すまっしゅ
            if 'すまっしゅ' in title:
                match = re.search(r'[　\s]+([０-９\d]+)', title)
                if match:
                    vol = match.group(1).translate(trans_table)
                    return f'僕のヒーローアカデミア_すまっしゅ_{vol}'
                return '僕のヒーローアカデミア_すまっしゅ_0'
            # スピンオフ: チームアップ
            if 'チームアップ' in title:
                match = re.search(r'[　\s]+([０-９\d]+)', title)
                if match:
                    vol = match.group(1).translate(trans_table)
                    return f'僕のヒーローアカデミア_チームアップ_{vol}'
                return '僕のヒーローアカデミア_チームアップ_0'
            # スピンオフ: 雄英白書
            if '雄英白書' in title:
                match = re.search(r'[　\s]+([０-９\d]+)', title)
                if match:
                    vol = match.group(1).translate(trans_table)
                    return f'僕のヒーローアカデミア_雄英白書_{vol}'
                return '僕のヒーローアカデミア_雄英白書_0'
            # 関連書籍
            if any(x in title for x in ['ＴＨＥ', 'かんたんイラスト', '公式キャラクター', 'ＴＶアニメ', 'カレンダー']):
                return '僕のヒーローアカデミア_関連書籍_0'
            # 本編
            match = re.search(r'僕のヒーローアカデミア　+([０-９\d]+)', title)
            if match:
                vol = match.group(1).translate(trans_table)
                return f'僕のヒーローアカデミア_original_{vol}'
            return '僕のヒーローアカデミア_original_0'
        # 8. 薬屋のひとりごと
        if title.startswith('薬屋のひとりごと') or title.startswith('特装版　薬屋のひとりごと'):
            # 画集
            if '画集' in title:
                return '薬屋のひとりごと_画集_0'
            # 特装版を除去
            title = re.sub(r'^特装版　', '', title)
            # サブシリーズ: 猫猫の後宮謎解き手
            if '～猫猫の後宮謎解き手' in title or '～猫猫の後' in title or '猫猫の後' in title:
                # 特装版を除去
                title = re.sub(r'　+特装版$', '', title)
                title = re.sub(r'　+(限定特装版|バリューパック)$', '', title)
                match = re.search(r'～猫猫の後[宮謎解き手]*　+([０-９\d]+)', title)
                if not match:
                    match = re.search(r'猫猫の後[宮謎解き手]*　+([０-９\d]+)', title)
                if match:
                    vol = match.group(1).translate(trans_table)
                    return f'薬屋のひとりごと_猫猫の後宮謎解き手_{vol}'
                return '薬屋のひとりごと_猫猫の後宮謎解き手_0'
            # バリューパック・特装版を除去
            title = re.sub(r'　+バリューパック$', '', title)
            title = re.sub(r'　+(限定特装版|特装版)$', '', title)
            # 本編
            match = re.search(r'薬屋のひとりごと　+([０-９\d]+)', title)
            if match:
                vol = match.group(1).translate(trans_table)
                return f'薬屋のひとりごと_original_{vol}'
            return '薬屋のひとりごと_original_0'
        # 9. ＳＰＹ×ＦＡＭＩＬＹ
        if 'ＳＰＹ×ＦＡＭＩＬＹ' in title:
            # 関連書籍
            if any(x in title for x in ['まんがノベライ', 'アニメーション', 'オペレーション', '家族の肖像',
                                        'フォージャー家', '公式ファンブック', 'ＴＶアニメ', 'カレンダー', '劇場版']):
                return 'ＳＰＹ×ＦＡＭＩＬＹ_関連書籍_0'
            # 特装版・同梱版を除去
            title = re.sub(r'　+(特装版|同梱版)$', '', title)
            # 本編
            match = re.search(r'ＳＰＹ×ＦＡＭＩＬＹ　+([０-９\d]+)', title)
            if match:
                vol = match.group(1).translate(trans_table)
                return f'ＳＰＹ×ＦＡＭＩＬＹ_original_{vol}'
            return 'ＳＰＹ×ＦＡＭＩＬＹ_original_0'
        # 10. キングダム
        if 'キングダム' in title:
            # 除外: 別作品
            exclude_keywords = ['ツキウタ', 'オレ様', 'アニア', '恐竜', '餃子', '迷宮',
                               'サキュバス', 'ラビッツ', '冒険大陸', 'アニマル']
            if any(x in title for x in exclude_keywords):
                return original_title
            # 完全版
            if '完全版' in title:
                match = re.search(r'[０-９\d]+', title)
                if match:
                    vol = match.group(0).translate(trans_table)
                    return f'キングダム_完全版_{vol}'
                return 'キングダム_完全版_0'
            # 映画
            if '映画' in title or '劇場版' in title or '大将軍の帰還' in title or '運命の炎' in title:
                return 'キングダム_映画_0'
            # 関連書籍
            if any(x in title for x in ['公式ガイド', '公式問題集', '英雄風雲録', '水晶玉子', 'ビジュアル', '英傑列紀']):
                return 'キングダム_関連書籍_0'
            # 本編（"キングダム　"で厳密にマッチング）
            if title.startswith('キングダム　'):
                match = re.search(r'キングダム　+([０-９\d]+)', title)
                if match:
                    vol = match.group(1).translate(trans_table)
                    return f'キングダム_original_{vol}'
            return 'キングダム_original_0'
        # 11. 【推しの子】
        if '【推しの子】' in title:
            # スピンオフ
            if '一番星のスピカ' in title:
                return '【推しの子】_一番星のスピカ_0'
            if '二人のエチュード' in title:
                return '【推しの子】_二人のエチュード_0'
            # 関連書籍
            if any(x in title for x in ['まんがノベライズ', 'カラーリング', 'イラスト集', '公式ガイ',
                                        '映画', 'セット', 'ＴＶアニメ', 'カレンダー']):
                return '【推しの子】_関連書籍_0'
            # 特装版を除去
            title = re.sub(r'　+ＳＰＥＣＩＡＬ.*$', '', title)
            # 本編
            match = re.search(r'【推しの子】　+([０-９\d]+)', title)
            if match:
                vol = match.group(1).translate(trans_table)
                return f'【推しの子】_original_{vol}'
            return '【推しの子】_original_0'
        # 12. カグラバチ
        if 'カグラバチ' in title:
            match = re.search(r'カグラバチ　+([０-９\d]+)', title)
            if match:
                vol = match.group(1).translate(trans_table)
                return f'カグラバチ_original_{vol}'
            return 'カグラバチ_original_0'
        # 13. 転生したらスライムだった件
        if '転生したらスライムだった件' in title:
            # スピンオフ: 転ちゅら
            if '転ちゅら' in title:
                match = re.search(r'[　\s]+([０-９\d]+)', title)
                if match:
                    vol = match.group(1).translate(trans_table)
                    return f'転生したらスライムだった件_転ちゅら_{vol}'
                return '転生したらスライムだった件_転ちゅら_0'
            # スピンオフ: クレイマ
            if 'クレイマ' in title:
                match = re.search(r'[　\s]+([０-９\d]+)', title)
                if match:
                    vol = match.group(1).translate(trans_table)
                    return f'転生したらスライムだった件_クレイマ_{vol}'
                return '転生したらスライムだった件_クレイマ_0'
            # スピンオフ: 異聞～魔国
            if '異聞～魔国' in title or '異聞　魔国' in title:
                match = re.search(r'[　\s]+([０-９\d]+)', title)
                if match:
                    vol = match.group(1).translate(trans_table)
                    return f'転生したらスライムだった件_異聞魔国_{vol}'
                return '転生したらスライムだった件_異聞魔国_0'
            # スピンオフ: ～魔物の国
            if '～魔物の国' in title:
                match = re.search(r'[　\s]+([０-９\d]+)', title)
                if match:
                    vol = match.group(1).translate(trans_table)
                    return f'転生したらスライムだった件_魔物の国_{vol}'
                return '転生したらスライムだった件_魔物の国_0'
            # スピンオフ: 美食伝
            if '美食伝' in title:
                match = re.search(r'[　\s]+([０-９\d]+)', title)
                if match:
                    vol = match.group(1).translate(trans_table)
                    return f'転生したらスライムだった件_美食伝_{vol}'
                return '転生したらスライムだった件_美食伝_0'
            # スピンオフ: 番外編
            if '番外編' in title:
                match = re.search(r'[　\s]+([０-９\d]+)', title)
                if match:
                    vol = match.group(1).translate(trans_table)
                    return f'転生したらスライムだった件_番外編_{vol}'
                return '転生したらスライムだった件_番外編_0'
            # 関連書籍
            if any(x in title for x in ['異世界サバイ', '転生漢字ドリ', 'で学べる', '公式キャラクタ', 'ＡＮＩＭＥ']):
                return '転生したらスライムだった件_関連書籍_0'
            # 特装版等を除去
            title = re.sub(r'　+(限定版|特装版|バリューパック)$', '', title)
            # 上中下巻
            match = re.search(r'転生したらスライムだった件　+([０-９\d\.]+)　+(上|中|下)', title)
            if match:
                vol = match.group(1).translate(trans_table)
                part = match.group(2)
                return f'転生したらスライムだった件_original_{vol}{part}'
            # 全３巻セット等
            if '全３巻' in title:
                match = re.search(r'[　\s]+([０-９\d]+)', title)
                if match:
                    vol = match.group(1).translate(trans_table)
                    return f'転生したらスライムだった件_original_{vol}'
            # 通常巻数
            match = re.search(r'転生したらスライムだった件　+([０-９\d\.]+)', title)
            if match:
                vol = match.group(1).translate(trans_table)
                return f'転生したらスライムだった件_original_{vol}'
            return '転生したらスライムだった件_original_0'
        # 14. ダンダダン
        if 'ダンダダン' in title:
            # 関連書籍
            if '超常現象解体新書' in title:
                return 'ダンダダン_関連書籍_0'
            # 本編
            match = re.search(r'ダンダダン　+([０-９\d]+)', title)
            if match:
                vol = match.group(1).translate(trans_table)
                return f'ダンダダン_original_{vol}'
            return 'ダンダダン_original_0'
        # 15. チェンソーマン
        if 'チェンソーマン' in title:
            # 関連書籍
            if any(x in title for x in ['バディ・ストーリーズ', 'ＴＶアニメ']):
                return 'チェンソーマン_関連書籍_0'
            match = re.search(r'チェンソーマン　+([０-９\d]+)', title)
            if match:
                vol = match.group(1).translate(trans_table)
                return f'チェンソーマン_original_{vol}'
            return 'チェンソーマン_original_0'
        # 16. アオのハコ
        if 'アオのハコ' in title:
            # Prologue
            if 'Ｐｒｏｌｏｇｕｅ' in title:
                return 'アオのハコ_Prologue_0'
            # 本編
            match = re.search(r'アオのハコ　+([０-９\d]+)', title)
            if match:
                vol = match.group(1).translate(trans_table)
                return f'アオのハコ_original_{vol}'
            return 'アオのハコ_original_0'
        # 17. 名探偵コナン
        if '名探偵コナン' in title:
            # 本編
            match = re.search(r'名探偵コナン　+([０-９\d]+)', title)
            if match:
                vol = match.group(1).translate(trans_table)
                return f'名探偵コナン_original_{vol}'
            # その他は全て関連
            return '名探偵コナン_関連書籍_0'
        # 18. ワンパンマン
        if 'ワンパンマン' in title:
            match = re.search(r'ワンパンマン　+([０-９\d]+)', title)
            if match:
                vol = match.group(1).translate(trans_table)
                return f'ワンパンマン_original_{vol}'
            return 'ワンパンマン_original_0'
        # 19. 地縛少年花子くん
        if '地縛少年花子くん' in title or '地縛少年　花子くん' in title:
            # 画集
            if '画集' in title:
                match = re.search(r'[　\s]+([０-９\d]+)', title)
                if match:
                    vol = match.group(1).translate(trans_table)
                    return f'地縛少年花子くん_画集_{vol}'
                return '地縛少年花子くん_画集_0'
            match = re.search(r'地縛少年[　]*花子くん　+([０-９\d]+)', title)
            if match:
                vol = match.group(1).translate(trans_table)
                return f'地縛少年花子くん_original_{vol}'
            return '地縛少年花子くん_original_0'
        # 20. マッシュル
        if 'マッシュル' in title:
            match = re.search(r'マッシュル[　\s]+([０-９\d]+)', title)
            if match:
                vol = match.group(1).translate(trans_table)
                return f'マッシュル_original_{vol}'
            return 'マッシュル_original_0'
        # 21. ＳＡＫＡＭＯＴＯ　ＤＡＹＳ
        if 'ＳＡＫＡＭＯＴＯ　ＤＡＹＳ' in title:
            # 関連書籍
            if any(x in title for x in ['殺し屋のメソ', '殺し屋ブルー']):
                return 'ＳＡＫＡＭＯＴＯ　ＤＡＹＳ_関連書籍_0'
            match = re.search(r'[　\s]+([０-９\d]+)', title)
            if match:
                vol = match.group(1).translate(trans_table)
                return f'ＳＡＫＡＭＯＴＯ　ＤＡＹＳ_original_{vol}'
            return 'ＳＡＫＡＭＯＴＯ　ＤＡＹＳ_original_0'
        # 22. ＨＵＮＴＥＲ×ＨＵＮＴＥＲ
        if 'ＨＵＮＴＥＲ×ＨＵＮＴＥＲ' in title:
            match = re.search(r'[　\s]+([０-９\d]+)', title)
            if match:
                vol = match.group(1).translate(trans_table)
                return f'ＨＵＮＴＥＲ×ＨＵＮＴＥＲ_original_{vol}'
            return 'ＨＵＮＴＥＲ×ＨＵＮＴＥＲ_original_0'
        # 23. 逃げ上手の若君
        if '逃げ上手の若君' in title:
            match = re.search(r'逃げ上手の若君　+([０-９\d]+)', title)
            if match:
                vol = match.group(1).translate(trans_table)
                return f'逃げ上手の若君_original_{vol}'
            return '逃げ上手の若君_original_0'
        # 24. 夜桜さんちの大作戦
        if '夜桜さんちの大作戦' in title:
            # 関連書籍
            if any(x in title for x in ['おるすばん', '観察日記']):
                return '夜桜さんちの大作戦_関連書籍_0'
            match = re.search(r'夜桜さんちの大作戦　+([０-９\d]+)', title)
            if match:
                vol = match.group(1).translate(trans_table)
                return f'夜桜さんちの大作戦_original_{vol}'
            return '夜桜さんちの大作戦_original_0'
        # 25. アオアシ
        if 'アオアシ' in title and 'アオのハコ' not in title:
            # ジュニア版
            if 'ジュニア版' in title:
                match = re.search(r'[　\s]+([０-９\d]+)', title)
                if match:
                    vol = match.group(1).translate(trans_table)
                    return f'アオアシ_ジュニア版_{vol}'
                return 'アオアシ_ジュニア版_0'
            # スピンオフ: ブラザーフット
            if 'ブラザーフット' in title:
                match = re.search(r'[　\s]+([０-９\d]+)', title)
                if match:
                    vol = match.group(1).translate(trans_table)
                    return f'アオアシ_ブラザーフット_{vol}'
                return 'アオアシ_ブラザーフット_0'
            # 小説
            if title.startswith('小説'):
                match = re.search(r'[　\s]+([０-９\d]+)', title)
                if match:
                    vol = match.group(1).translate(trans_table)
                    return f'アオアシ_小説_{vol}'
                return 'アオアシ_小説_0'
            # 関連書籍
            if 'に学ぶ' in title:
                return 'アオアシ_関連書籍_0'
            # 本編
            match = re.search(r'アオアシ　+([０-９\d]+)', title)
            if match:
                vol = match.group(1).translate(trans_table)
                return f'アオアシ_original_{vol}'
            return 'アオアシ_original_0'
        # 26. つかめ！理科ダマン
        if 'つかめ！理科ダマン' in title:
            match = re.search(r'[　\s]+([０-９\d]+)', title)
            if match:
                vol = match.group(1).translate(trans_table)
                return f'つかめ！理科ダマン_original_{vol}'
            return 'つかめ！理科ダマン_original_0'
        # 27. ゴールデンカムイ
        if 'ゴールデンカムイ' in title:
            # 関連書籍
            if any(x in title for x in ['アイヌ文化', '絵から学ぶ', '公式フ', '映画']):
                return 'ゴールデンカムイ_関連書籍_0'
            # 本編
            match = re.search(r'ゴールデンカムイ　+([０-９\d]+)', title)
            if match:
                vol = match.group(1).translate(trans_table)
                return f'ゴールデンカムイ_original_{vol}'
            return 'ゴールデンカムイ_original_0'
        # 28. 忘却バッテリー
        if '忘却バッテリー' in title:
            match = re.search(r'忘却バッテリー　+([０-９\d]+)', title)
            if match:
                vol = match.group(1).translate(trans_table)
                return f'忘却バッテリー_original_{vol}'
            return '忘却バッテリー_original_0'
        # 29. シャングリラ・フロンティア
        if 'シャングリラ・フロンティア' in title:
            # 関連書籍
            if 'るるぶ' in title:
                return 'シャングリラ・フロンティア_関連書籍_0'
            # 特装版・限定版を除去
            title = re.sub(r'　+(特装版|限定版)$', '', title)
            # クソゲーハンター版
            if '～クソゲ' in title or 'クソゲーハンター' in title:
                match = re.search(r'[　\s]+([０-９\d]+)', title)
                if match:
                    vol = match.group(1).translate(trans_table)
                    return f'シャングリラ・フロンティア_クソゲーハンター_{vol}'
                return 'シャングリラ・フロンティア_クソゲーハンター_0'
            # 通常版
            match = re.search(r'シャングリラ・フロンティア　+([０-９\d]+)', title)
            if match:
                vol = match.group(1).translate(trans_table)
                return f'シャングリラ・フロンティア_original_{vol}'
            return 'シャングリラ・フロンティア_original_0'
        # 30. 時々ボソッとロシア語でデレる隣のアー
        if '時々ボソッとロシア語でデレる隣のアー' in title:
            match = re.search(r'[　\s]+([０-９\d]+)', title)
            if match:
                vol = match.group(1).translate(trans_table)
                return f'時々ボソッとロシア語でデレる隣のアーリャさん_original_{vol}'
            return '時々ボソッとロシア語でデレる隣のアーリャさん_original_0'
        # 31. 大ピンチずかん
        if '大ピンチずかん' in title:
            match = re.search(r'[　\s]+([０-９\d]+)', title)
            if match:
                vol = match.group(1).translate(trans_table)
                return f'大ピンチずかん_original_{vol}'
            return '大ピンチずかん_original_0'
        # 32. 僕の心のヤバイやつ
        if '僕の心のヤバイやつ' in title:
            # 特装版を除去
            title = re.sub(r'^特装版　', '', title)
            # 小説
            if title.startswith('小説'):
                return '僕の心のヤバイやつ_小説_0'
            # 関連書籍
            if 'ＴＶアニメ' in title:
                return '僕の心のヤバイやつ_関連書籍_0'
            match = re.search(r'僕の心のヤバイやつ　+([０-９\d]+)', title)
            if match:
                vol = match.group(1).translate(trans_table)
                return f'僕の心のヤバイやつ_original_{vol}'
            return '僕の心のヤバイやつ_original_0'
        # 33. ダンジョン飯
        if 'ダンジョン飯' in title:
            # 関連書籍
            if any(x in title for x in ['ワールドガイド', '冒険者バイ', '英会話', 'Ｗａｌｋｅｒ']):
                return 'ダンジョン飯_関連書籍_0'
            # 本編
            match = re.search(r'ダンジョン飯　+([０-９\d]+)', title)
            if match:
                vol = match.group(1).translate(trans_table)
                return f'ダンジョン飯_original_{vol}'
            return 'ダンジョン飯_original_0'
        # 34. ＷＩＮＤ　ＢＲＥＡＫＥＲ
        if 'ＷＩＮＤ　ＢＲＥＡＫＥＲ' in title:
            # 関連書籍
            if '公式キャラクタ' in title:
                return 'ＷＩＮＤ　ＢＲＥＡＫＥＲ_関連書籍_0'
            # 本編
            match = re.search(r'[　\s]+([０-９\d]+)', title)
            if match:
                vol = match.group(1).translate(trans_table)
                return f'ＷＩＮＤ　ＢＲＥＡＫＥＲ_original_{vol}'
            return 'ＷＩＮＤ　ＢＲＥＡＫＥＲ_original_0'
        # 35. 魔入りました
        if '魔入りました' in title:
            # 小説
            if title.startswith('小説'):
                match = re.search(r'[　\s]+([０-９\d]+)', title)
                if match:
                    vol = match.group(1).translate(trans_table)
                    return f'魔入りました_小説_{vol}'
                return '魔入りました_小説_0'
            # スピンオフ
            if 'ｉｆ　Ｅｐｉ' in title:
                match = re.search(r'[　\s]+([０-９\d]+)', title)
                if match:
                    vol = match.group(1).translate(trans_table)
                    return f'魔入りました_ifEpi_{vol}'
                return '魔入りました_ifEpi_0'
            if '外伝' in title:
                match = re.search(r'[　\s]+([０-９\d]+)', title)
                if match:
                    vol = match.group(1).translate(trans_table)
                    return f'魔入りました_外伝_{vol}'
                return '魔入りました_外伝_0'
            # 関連書籍
            if any(x in title for x in ['スターターＢＯＸ', '公式アンソロ', '公式ファンブック', 'で学ぶ']):
                return '魔入りました_関連書籍_0'
            # 本編
            match = re.search(r'魔入りました！入間くん　+([０-９\d]+)', title)
            if match:
                vol = match.group(1).translate(trans_table)
                return f'魔入りました_original_{vol}'
            return '魔入りました_original_0'
        # 36. ハイキュー
        if 'ハイキュー' in title:
            # スピンオフ: れっつ！ハイキュー！？
            if 'れっつ！ハイキュー！？' in title:
                match = re.search(r'[　\s]+([０-９\d]+)', title)
                if match:
                    vol = match.group(1).translate(trans_table)
                    return f'ハイキュー_れっつ_{vol}'
                return 'ハイキュー_れっつ_0'
            # スピンオフ: ハイキュー部！！
            if 'ハイキュー部！！' in title:
                match = re.search(r'[　\s]+([０-９\d]+)', title)
                if match:
                    vol = match.group(1).translate(trans_table)
                    return f'ハイキュー_ハイキュー部_{vol}'
                return 'ハイキュー_ハイキュー部_0'
            # 小説版
            if 'ショーセツバン' in title:
                match = re.search(r'[　\s]+([０-９\d]+)', title)
                if match:
                    vol = match.group(1).translate(trans_table)
                    return f'ハイキュー_小説_{vol}'
                return 'ハイキュー_小説_0'
            # 関連書籍
            if any(x in title for x in ['ファイナルガイド', 'クロニクル', 'セイシュンメイカン', 'カラーイラスト',
                                        'Ｍａｇａｚｉｎ', 'ＴＶアニメチームブック', '劇場版', '排球本', '生き方がラク',
                                        'カレンダー', 'Ｃｏｍｐｌｅｔｅ']):
                return 'ハイキュー_関連書籍_0'
            # 本編
            match = re.search(r'ハイキュー！！　+([０-９\d]+)', title)
            if match:
                vol = match.group(1).translate(trans_table)
                return f'ハイキュー_original_{vol}'
            return 'ハイキュー_original_0'
        # 37. 黄泉のツガイ
        if '黄泉のツガイ' in title:
            # 特装版を除去
            title = re.sub(r'^特装版　', '', title)
            match = re.search(r'黄泉のツガイ　+([０-９\d]+)', title)
            if match:
                vol = match.group(1).translate(trans_table)
                return f'黄泉のツガイ_original_{vol}'
            return '黄泉のツガイ_original_0'
        # 38. 終末のワルキューレ
        if '終末のワルキューレ' in title:
            # スピンオフ: 奇譚　ジャック・ザ
            if '奇譚' in title and 'ジャック' in title:
                match = re.search(r'[　\s]+([０-９\d]+)', title)
                if match:
                    vol = match.group(1).translate(trans_table)
                    return f'終末のワルキューレ_奇譚ジャックザ_{vol}'
                return '終末のワルキューレ_奇譚ジャックザ_0'
            # スピンオフ: 禁伝　神々の黙示録
            if '禁伝' in title:
                match = re.search(r'[　\s]+([０-９\d]+)', title)
                if match:
                    vol = match.group(1).translate(trans_table)
                    return f'終末のワルキューレ_禁伝_{vol}'
                return '終末のワルキューレ_禁伝_0'
            # スピンオフ: 異聞　呂布奉先飛将
            if '異聞' in title:
                match = re.search(r'[　\s]+([０-９\d]+)', title)
                if match:
                    vol = match.group(1).translate(trans_table)
                    return f'終末のワルキューレ_異聞呂布奉先飛将_{vol}'
                return '終末のワルキューレ_異聞呂布奉先飛将_0'
            # Special版を除去
            title = re.sub(r'　+Ｓｐｅｃｉａｌ.*$', '', title)
            title = re.sub(r'　+ＳｐｅｃｉａｌＥ.*$', '', title)
            # 本編
            match = re.search(r'終末のワルキューレ　+([０-９\d]+)', title)
            if match:
                vol = match.group(1).translate(trans_table)
                return f'終末のワルキューレ_original_{vol}'
            return '終末のワルキューレ_original_0'
        # 39. ちいかわ
        if 'ちいかわ' in title:
            # 本編: なんか小さくてかわいいやつ
            if 'なんか小さくてかわいいやつ' in title or 'なんか小さくてかわ' in title:
                # 特装版を除去
                title = re.sub(r'^特装版　', '', title)
                title = re.sub(r'　+特装版$', '', title)
                match = re.search(r'[　\s]+([０-９\d]+)', title)
                if match:
                    vol = match.group(1).translate(trans_table)
                    return f'ちいかわ_original_{vol}'
                return 'ちいかわ_original_0'
            # その他は全て関連書籍
            return 'ちいかわ_関連書籍_0'
        # 40. 光が死んだ夏
        if '光が死んだ夏' in title:
            # 特装版を除去
            title = re.sub(r'　+特装版$', '', title)
            match = re.search(r'光が死んだ夏　+([０-９\d]+)', title)
            if match:
                vol = match.group(1).translate(trans_table)
                return f'光が死んだ夏_original_{vol}'
            return '光が死んだ夏_original_0'
        # 41. 助太刀稼業
        if '助太刀稼業' in title:
            match = re.search(r'助太刀稼業　+([０-９\d]+)', title)
            if match:
                vol = match.group(1).translate(trans_table)
                return f'助太刀稼業_original_{vol}'
            return '助太刀稼業_original_0'
        # 42. 金色のガッシュ！！２
        if '金色のガッシュ！！２' in title:
            match = re.search(r'[　\s]+([０-９\d]+)', title)
            if match:
                vol = match.group(1).translate(trans_table)
                return f'金色のガッシュ２_original_{vol}'
            return '金色のガッシュ２_original_0'
        # 43. スーパーの裏でヤニ吸うふたり
        if 'スーパーの裏でヤニ吸うふたり' in title:
            # 特装版を除去
            title = re.sub(r'^特装版　', '', title)
            match = re.search(r'[　\s]+([０-９\d]+)', title)
            if match:
                vol = match.group(1).translate(trans_table)
                return f'スーパーの裏でヤニ吸うふたり_original_{vol}'
            return 'スーパーの裏でヤニ吸うふたり_original_0'
        # 44. わたしの幸せな結婚
        if 'わたしの幸せな結婚' in title:
            # 画集
            if '画集' in title:
                return 'わたしの幸せな結婚_画集_0'
            # 特装版・Blu-ray付を除去
            title = re.sub(r'^特装版　', '', title)
            title = re.sub(r'　+Ｂｌｕ－ｒａｙ付.*$', '', title)
            # 本編
            match = re.search(r'[　\s]+([０-９\d]+)', title)
            if match:
                vol = match.group(1).translate(trans_table)
                return f'わたしの幸せな結婚_original_{vol}'
            return 'わたしの幸せな結婚_original_0'
        # 45. その着せ替え人形は恋をする
        if 'その着せ替え人形は恋をする' in title:
            # TVアニメ関連
            if 'ＴＶアニメ' in title:
                return 'その着せ替え人形は恋をする_関連書籍_0'
            # 特装版を除去
            title = re.sub(r'^特装版　', '', title)
            # 本編
            match = re.search(r'[　\s]+([０-９\d]+)', title)
            if match:
                vol = match.group(1).translate(trans_table)
                return f'その着せ替え人形は恋をする_original_{vol}'
            return 'その着せ替え人形は恋をする_original_0'
        # 46. 片田舎のおっさん、剣聖になる
        if '片田舎のおっさん、剣聖になる' in title:
            match = re.search(r'[　\s]+([０-９\d]+)', title)
            if match:
                vol = match.group(1).translate(trans_table)
                return f'片田舎のおっさん、剣聖になる_original_{vol}'
            return '片田舎のおっさん、剣聖になる_original_0'
        # 47. ＢＬＵＥ　ＧＩＡＮＴ　ＭＯＭＥＮＴＵ
        if 'ＢＬＵＥ　ＧＩＡＮＴ　ＭＯＭＥＮＴＵ' in title:
            match = re.search(r'[　\s]+([０-９\d]+)', title)
            if match:
                vol = match.group(1).translate(trans_table)
                return f'ＢＬＵＥ　ＧＩＡＮＴ　ＭＯＭＥＮＴＵ_original_{vol}'
            return 'ＢＬＵＥ　ＧＩＡＮＴ　ＭＯＭＥＮＴＵ_original_0'
        # 48. 香君
        if '香君' in title:
            # 小説版（上下）
            if '上' in title or '下' in title:
                return '香君_小説_0'
            # 本編
            match = re.search(r'香君　+([０-９\d]+)', title)
            if match:
                vol = match.group(1).translate(trans_table)
                return f'香君_original_{vol}'
            return '香君_original_0'
        # 49. 春の嵐とモンスター
        if '春の嵐とモンスター' in title:
            match = re.search(r'[　\s]+([０-９\d]+)', title)
            if match:
                vol = match.group(1).translate(trans_table)
                return f'春の嵐とモンスター_original_{vol}'
            return '春の嵐とモンスター_original_0'
        # 50. 恋せよまやかし天使ども
        if '恋せよまやかし天使ども' in title:
            match = re.search(r'[　\s]+([０-９\d]+)', title)
            if match:
                vol = match.group(1).translate(trans_table)
                return f'恋せよまやかし天使ども_original_{vol}'
            return '恋せよまやかし天使ども_original_0'
        # 51. ファントムバスターズ
        if 'ファントムバスターズ' in title:
            match = re.search(r'[　\s]+([０-９\d]+)', title)
            if match:
                vol = match.group(1).translate(trans_table)
                return f'ファントムバスターズ_original_{vol}'
            return 'ファントムバスターズ_original_0'
        # 52. 鬼の花嫁
        if '鬼の花嫁' in title:
            # スピンオフ: は喰べられたい
            if '喰べられたい' in title:
                match = re.search(r'[　\s]+([０-９\d]+)', title)
                if match:
                    vol = match.group(1).translate(trans_table)
                    return f'鬼の花嫁_喰べられたい_{vol}'
                return '鬼の花嫁_喰べられたい_0'
            # サブシリーズ: 新婚編
            if '新婚編' in title:
                match = re.search(r'新婚編　+([０-９\d]+)', title)
                if match:
                    vol = match.group(1).translate(trans_table)
                    return f'鬼の花嫁_新婚編_{vol}'
                return '鬼の花嫁_新婚編_0'
            # 特装版を除去
            title = re.sub(r'　+特装版$', '', title)
            # サブタイトルを除去
            title = re.sub(r'～.*～$', '', title)
            title = re.sub(r'～.*$', '', title)
            # 本編
            match = re.search(r'鬼の花嫁　+([０-９\d]+)', title)
            if match:
                vol = match.group(1).translate(trans_table)
                return f'鬼の花嫁_original_{vol}'
            return '鬼の花嫁_original_0'
        # 53. チ。－地球の運動について－
        if 'チ。－地球の運動について－' in title:
            title = re.sub(r'^特装版　', '', title)
            match = re.search(r'[０-９\d]+', title)
            if match:
                vol = match.group(0).translate(trans_table)
                return f'チ。－地球の運動について－_original_{vol}'
            return 'チ。－地球の運動について－_original_0'
        # 54. ドラゴンボール超
        if 'ドラゴンボール超' in title:
            title = re.sub(r'^特装版　', '', title)
            match = re.search(r'[０-９\d]+', title)
            if match:
                vol = match.group(0).translate(trans_table)
                return f'ドラゴンボール超_original_{vol}'
            return 'ドラゴンボール超_original_0'
        # 55. Ｄｒ．ＳＴＯＮＥ
        if 'Ｄｒ．ＳＴＯＮＥ' in title:
            title = re.sub(r'^特装版　', '', title)
            match = re.search(r'[０-９\d]+', title)
            if match:
                vol = match.group(0).translate(trans_table)
                return f'Ｄｒ．ＳＴＯＮＥ_original_{vol}'
            return 'Ｄｒ．ＳＴＯＮＥ_original_0'
        # 56. 古見さんは、コミュ症です。
        if '古見さんは、コミュ症です。' in title:
            title = re.sub(r'^特装版　', '', title)
            match = re.search(r'[０-９\d]+', title)
            if match:
                vol = match.group(0).translate(trans_table)
                return f'古見さんは、コミュ症です。_original_{vol}'
            return '古見さんは、コミュ症です。_original_0'
        # 57. 幼稚園ＷＡＲＳ
        if '幼稚園ＷＡＲＳ' in title:
            title = re.sub(r'^特装版　', '', title)
            match = re.search(r'[０-９\d]+', title)
            if match:
                vol = match.group(0).translate(trans_table)
                return f'幼稚園ＷＡＲＳ_original_{vol}'
            return '幼稚園ＷＡＲＳ_original_0'
        # 58. 魔界の主役は我々だ！
        if '魔界の主役は我々だ！' in title:
            title = re.sub(r'^特装版　', '', title)
            match = re.search(r'[０-９\d]+', title)
            if match:
                vol = match.group(0).translate(trans_table)
                return f'魔界の主役は我々だ！_original_{vol}'
            return '魔界の主役は我々だ！_original_0'
        # 59. ミステリと言う勿れ
        if 'ミステリと言う勿れ' in title:
            title = re.sub(r'^特装版　', '', title)
            match = re.search(r'[０-９\d]+', title)
            if match:
                vol = match.group(0).translate(trans_table)
                return f'ミステリと言う勿れ_original_{vol}'
            return 'ミステリと言う勿れ_original_0'
        # 60. ワールドトリガー
        if 'ワールドトリガー' in title:
            title = re.sub(r'^特装版　', '', title)
            match = re.search(r'[０-９\d]+', title)
            if match:
                vol = match.group(0).translate(trans_table)
                return f'ワールドトリガー_original_{vol}'
            return 'ワールドトリガー_original_0'
        # 61. ＤＲＡＧＯＮ　ＢＡＬＬ
        if 'ＤＲＡＧＯＮ　ＢＡＬＬ' in title:
            title = re.sub(r'^特装版　', '', title)
            match = re.search(r'[０-９\d]+', title)
            if match:
                vol = match.group(0).translate(trans_table)
                return f'ＤＲＡＧＯＮ　ＢＡＬＬ_original_{vol}'
            return 'ＤＲＡＧＯＮ　ＢＡＬＬ_original_0'
        # 62. ＭＦゴースト
        if 'ＭＦゴースト' in title:
            title = re.sub(r'^特装版　', '', title)
            match = re.search(r'[０-９\d]+', title)
            if match:
                vol = match.group(0).translate(trans_table)
                return f'ＭＦゴースト_original_{vol}'
            return 'ＭＦゴースト_original_0'
        # 63. 氷の城壁
        if '氷の城壁' in title:
            title = re.sub(r'^特装版　', '', title)
            match = re.search(r'[０-９\d]+', title)
            if match:
                vol = match.group(0).translate(trans_table)
                return f'氷の城壁_original_{vol}'
            return '氷の城壁_original_0'
        # 64. よふかしのうた
        if 'よふかしのうた' in title:
            title = re.sub(r'^特装版　', '', title)
            match = re.search(r'[０-９\d]+', title)
            if match:
                vol = match.group(0).translate(trans_table)
                return f'よふかしのうた_original_{vol}'
            return 'よふかしのうた_original_0'
        # 65. 黙示録の四騎士
        if '黙示録の四騎士' in title:
            title = re.sub(r'^特装版　', '', title)
            match = re.search(r'[０-９\d]+', title)
            if match:
                vol = match.group(0).translate(trans_table)
                return f'黙示録の四騎士_original_{vol}'
            return '黙示録の四騎士_original_0'
        # 66. ダイヤモンドの功罪
        if 'ダイヤモンドの功罪' in title:
            title = re.sub(r'^特装版　', '', title)
            match = re.search(r'[０-９\d]+', title)
            if match:
                vol = match.group(0).translate(trans_table)
                return f'ダイヤモンドの功罪_original_{vol}'
            return 'ダイヤモンドの功罪_original_0'
        # 67. ハニーレモンソーダ
        if 'ハニーレモンソーダ' in title:
            title = re.sub(r'^特装版　', '', title)
            match = re.search(r'[０-９\d]+', title)
            if match:
                vol = match.group(0).translate(trans_table)
                return f'ハニーレモンソーダ_original_{vol}'
            return 'ハニーレモンソーダ_original_0'
        # 68. ゆびさきと恋々
        if 'ゆびさきと恋々' in title:
            title = re.sub(r'^特装版　', '', title)
            match = re.search(r'[０-９\d]+', title)
            if match:
                vol = match.group(0).translate(trans_table)
                return f'ゆびさきと恋々_original_{vol}'
            return 'ゆびさきと恋々_original_0'
        # 69. ウマ娘　シンデレラグレイ
        if 'ウマ娘　シンデレラグレイ' in title:
            title = re.sub(r'^特装版　', '', title)
            match = re.search(r'[０-９\d]+', title)
            if match:
                vol = match.group(0).translate(trans_table)
                return f'ウマ娘　シンデレラグレイ_original_{vol}'
            return 'ウマ娘　シンデレラグレイ_original_0'
        # 70. 山田くんとＬｖ９９９の恋をする
        if '山田くんとＬｖ９９９の恋をする' in title:
            title = re.sub(r'^特装版　', '', title)
            match = re.search(r'[０-９\d]+', title)
            if match:
                vol = match.group(0).translate(trans_table)
                return f'山田くんとＬｖ９９９の恋をする_original_{vol}'
            return '山田くんとＬｖ９９９の恋をする_original_0'
        # 71. 進撃の巨人
        if '進撃の巨人' in title:
            title = re.sub(r'^特装版　', '', title)
            match = re.search(r'[０-９\d]+', title)
            if match:
                vol = match.group(0).translate(trans_table)
                return f'進撃の巨人_original_{vol}'
            return '進撃の巨人_original_0'
        # 72. あかね噺
        if 'あかね噺' in title:
            title = re.sub(r'^特装版　', '', title)
            match = re.search(r'[０-９\d]+', title)
            if match:
                vol = match.group(0).translate(trans_table)
                return f'あかね噺_original_{vol}'
            return 'あかね噺_original_0'
        # 73. ブルーピリオド
        if 'ブルーピリオド' in title:
            title = re.sub(r'^特装版　', '', title)
            match = re.search(r'[０-９\d]+', title)
            if match:
                vol = match.group(0).translate(trans_table)
                return f'ブルーピリオド_original_{vol}'
            return 'ブルーピリオド_original_0'
        # 74. 初×婚
        if '初×婚' in title:
            title = re.sub(r'^特装版　', '', title)
            match = re.search(r'[０-９\d]+', title)
            if match:
                vol = match.group(0).translate(trans_table)
                return f'初×婚_original_{vol}'
            return '初×婚_original_0'
        # 75. はたらく細胞
        if 'はたらく細胞' in title:
            title = re.sub(r'^特装版　', '', title)
            match = re.search(r'[０-９\d]+', title)
            if match:
                vol = match.group(0).translate(trans_table)
                return f'はたらく細胞_original_{vol}'
            return 'はたらく細胞_original_0'
        # 76. 文豪ストレイドッグス
        if '文豪ストレイドッグス' in title:
            title = re.sub(r'^特装版　', '', title)
            match = re.search(r'[０-９\d]+', title)
            if match:
                vol = match.group(0).translate(trans_table)
                return f'文豪ストレイドッグス_original_{vol}'
            return '文豪ストレイドッグス_original_0'
        # 77. まんがで！にゃんこ大戦争
        if 'まんがで！にゃんこ大戦争' in title:
            title = re.sub(r'^特装版　', '', title)
            match = re.search(r'[０-９\d]+', title)
            if match:
                vol = match.group(0).translate(trans_table)
                return f'まんがで！にゃんこ大戦争_original_{vol}'
            return 'まんがで！にゃんこ大戦争_original_0'
        # 78. ホタルの嫁入り
        if 'ホタルの嫁入り' in title:
            title = re.sub(r'^特装版　', '', title)
            match = re.search(r'[０-９\d]+', title)
            if match:
                vol = match.group(0).translate(trans_table)
                return f'ホタルの嫁入り_original_{vol}'
            return 'ホタルの嫁入り_original_0'
        # 79. うるわしの宵の月
        if 'うるわしの宵の月' in title:
            title = re.sub(r'^特装版　', '', title)
            match = re.search(r'[０-９\d]+', title)
            if match:
                vol = match.group(0).translate(trans_table)
                return f'うるわしの宵の月_original_{vol}'
            return 'うるわしの宵の月_original_0'
        # 80. 魔都精兵のスレイブ
        if '魔都精兵のスレイブ' in title:
            title = re.sub(r'^特装版　', '', title)
            match = re.search(r'[０-９\d]+', title)
            if match:
                vol = match.group(0).translate(trans_table)
                return f'魔都精兵のスレイブ_original_{vol}'
            return '魔都精兵のスレイブ_original_0'
        # 81. 東京卍リベンジャーズ
        if '東京卍リベンジャーズ' in title:
            title = re.sub(r'^特装版　', '', title)
            match = re.search(r'[０-９\d]+', title)
            if match:
                vol = match.group(0).translate(trans_table)
                return f'東京卍リベンジャーズ_original_{vol}'
            return '東京卍リベンジャーズ_original_0'
        # 82. 平和の国の島崎へ
        if '平和の国の島崎へ' in title:
            title = re.sub(r'^特装版　', '', title)
            match = re.search(r'[０-９\d]+', title)
            if match:
                vol = match.group(0).translate(trans_table)
                return f'平和の国の島崎へ_original_{vol}'
            return '平和の国の島崎へ_original_0'
        # 83. キメツ学園！
        if 'キメツ学園！' in title:
            title = re.sub(r'^特装版　', '', title)
            match = re.search(r'[０-９\d]+', title)
            if match:
                vol = match.group(0).translate(trans_table)
                return f'キメツ学園！_original_{vol}'
            return 'キメツ学園！_original_0'
        # 84. 男はつらいよＤＶＤコレクション全国
        if '男はつらいよＤＶＤコレクション全国' in title:
            title = re.sub(r'^特装版　', '', title)
            match = re.search(r'[０-９\d]+', title)
            if match:
                vol = match.group(0).translate(trans_table)
                return f'男はつらいよＤＶＤコレクション全国_original_{vol}'
            return '男はつらいよＤＶＤコレクション全国_original_0'
        # 85. ラーメン赤猫
        if 'ラーメン赤猫' in title:
            title = re.sub(r'^特装版　', '', title)
            match = re.search(r'[０-９\d]+', title)
            if match:
                vol = match.group(0).translate(trans_table)
                return f'ラーメン赤猫_original_{vol}'
            return 'ラーメン赤猫_original_0'
        # 86. ウィッチウォッチ
        if 'ウィッチウォッチ' in title:
            title = re.sub(r'^特装版　', '', title)
            match = re.search(r'[０-９\d]+', title)
            if match:
                vol = match.group(0).translate(trans_table)
                return f'ウィッチウォッチ_original_{vol}'
            return 'ウィッチウォッチ_original_0'
        # 87. 超人Ｘ
        if '超人Ｘ' in title:
            title = re.sub(r'^特装版　', '', title)
            match = re.search(r'[０-９\d]+', title)
            if match:
                vol = match.group(0).translate(trans_table)
                return f'超人Ｘ_original_{vol}'
            return '超人Ｘ_original_0'
        # 88. アンデッドアンラック
        if 'アンデッドアンラック' in title:
            title = re.sub(r'^特装版　', '', title)
            match = re.search(r'[０-９\d]+', title)
            if match:
                vol = match.group(0).translate(trans_table)
                return f'アンデッドアンラック_original_{vol}'
            return 'アンデッドアンラック_original_0'
        # 89. 弱虫ペダル
        if '弱虫ペダル' in title:
            title = re.sub(r'^特装版　', '', title)
            match = re.search(r'[０-９\d]+', title)
            if match:
                vol = match.group(0).translate(trans_table)
                return f'弱虫ペダル_original_{vol}'
            return '弱虫ペダル_original_0'
        # 90. 彼女、お借りします
        if '彼女、お借りします' in title:
            title = re.sub(r'^特装版　', '', title)
            match = re.search(r'[０-９\d]+', title)
            if match:
                vol = match.group(0).translate(trans_table)
                return f'彼女、お借りします_original_{vol}'
            return '彼女、お借りします_original_0'
        # 91. 杖と剣のウィストリア
        if '杖と剣のウィストリア' in title:
            title = re.sub(r'^特装版　', '', title)
            match = re.search(r'[０-９\d]+', title)
            if match:
                vol = match.group(0).translate(trans_table)
                return f'杖と剣のウィストリア_original_{vol}'
            return '杖と剣のウィストリア_original_0'
        # 92. ドラえもん
        if 'ドラえもん' in title:
            title = re.sub(r'^特装版　', '', title)
            match = re.search(r'[０-９\d]+', title)
            if match:
                vol = match.group(0).translate(trans_table)
                return f'ドラえもん_original_{vol}'
            return 'ドラえもん_original_0'
        # 93. ２．５次元の誘惑
        if '２．５次元の誘惑' in title:
            title = re.sub(r'^特装版　', '', title)
            match = re.search(r'[０-９\d]+', title)
            if match:
                vol = match.group(0).translate(trans_table)
                return f'２．５次元の誘惑_original_{vol}'
            return '２．５次元の誘惑_original_0'
        # 94. 暁のヨナ
        if '暁のヨナ' in title:
            title = re.sub(r'^特装版　', '', title)
            match = re.search(r'[０-９\d]+', title)
            if match:
                vol = match.group(0).translate(trans_table)
                return f'暁のヨナ_original_{vol}'
            return '暁のヨナ_original_0'
        # 95. ピンクとハバネロ
        if 'ピンクとハバネロ' in title:
            title = re.sub(r'^特装版　', '', title)
            match = re.search(r'[０-９\d]+', title)
            if match:
                vol = match.group(0).translate(trans_table)
                return f'ピンクとハバネロ_original_{vol}'
            return 'ピンクとハバネロ_original_0'
        # 96. 女の園の星
        if '女の園の星' in title:
            title = re.sub(r'^特装版　', '', title)
            match = re.search(r'[０-９\d]+', title)
            if match:
                vol = match.group(0).translate(trans_table)
                return f'女の園の星_original_{vol}'
            return '女の園の星_original_0'
        # 97. ぼっち・ざ・ろっく！
        if 'ぼっち・ざ・ろっく！' in title:
            title = re.sub(r'^特装版　', '', title)
            match = re.search(r'[０-９\d]+', title)
            if match:
                vol = match.group(0).translate(trans_table)
                return f'ぼっち・ざ・ろっく！_original_{vol}'
            return 'ぼっち・ざ・ろっく！_original_0'
        # 98. 九条の大罪
        if '九条の大罪' in title:
            title = re.sub(r'^特装版　', '', title)
            match = re.search(r'[０-９\d]+', title)
            if match:
                vol = match.group(0).translate(trans_table)
                return f'九条の大罪_original_{vol}'
            return '九条の大罪_original_0'
        # 99. 黒執事
        if '黒執事' in title:
            title = re.sub(r'^特装版　', '', title)
            match = re.search(r'[０-９\d]+', title)
            if match:
                vol = match.group(0).translate(trans_table)
                return f'黒執事_original_{vol}'
            return '黒執事_original_0'
        # 100. 魔法少女にあこがれて
        if '魔法少女にあこがれて' in title:
            title = re.sub(r'^特装版　', '', title)
            match = re.search(r'[０-９\d]+', title)
            if match:
                vol = match.group(0).translate(trans_table)
                return f'魔法少女にあこがれて_original_{vol}'
            return '魔法少女にあこがれて_original_0'
        # 101. ゆるキャン△
        if 'ゆるキャン△' in title:
            title = re.sub(r'^特装版　', '', title)
            match = re.search(r'[０-９\d]+', title)
            if match:
                vol = match.group(0).translate(trans_table)
                return f'ゆるキャン△_original_{vol}'
            return 'ゆるキャン△_original_0'
        # 102. 宇宙兄弟
        if '宇宙兄弟' in title:
            title = re.sub(r'^特装版　', '', title)
            match = re.search(r'[０-９\d]+', title)
            if match:
                vol = match.group(0).translate(trans_table)
                return f'宇宙兄弟_original_{vol}'
            return '宇宙兄弟_original_0'
        # 103. 桃源暗鬼
        if '桃源暗鬼' in title:
            title = re.sub(r'^特装版　', '', title)
            match = re.search(r'[０-９\d]+', title)
            if match:
                vol = match.group(0).translate(trans_table)
                return f'桃源暗鬼_original_{vol}'
            return '桃源暗鬼_original_0'
        # 104. 宝石の国
        if '宝石の国' in title:
            title = re.sub(r'^特装版　', '', title)
            match = re.search(r'[０-９\d]+', title)
            if match:
                vol = match.group(0).translate(trans_table)
                return f'宝石の国_original_{vol}'
            return '宝石の国_original_0'
        # 105. 青の祓魔師
        if '青の祓魔師' in title:
            title = re.sub(r'^特装版　', '', title)
            match = re.search(r'[０-９\d]+', title)
            if match:
                vol = match.group(0).translate(trans_table)
                return f'青の祓魔師_original_{vol}'
            return '青の祓魔師_original_0'
        # 106. 合コンに行ったら女がいなかった話
        if '合コンに行ったら女がいなかった話' in title:
            title = re.sub(r'^特装版　', '', title)
            match = re.search(r'[０-９\d]+', title)
            if match:
                vol = match.group(0).translate(trans_table)
                return f'合コンに行ったら女がいなかった話_original_{vol}'
            return '合コンに行ったら女がいなかった話_original_0'
        # 107. スキップとローファー
        if 'スキップとローファー' in title:
            title = re.sub(r'^特装版　', '', title)
            match = re.search(r'[０-９\d]+', title)
            if match:
                vol = match.group(0).translate(trans_table)
                return f'スキップとローファー_original_{vol}'
            return 'スキップとローファー_original_0'
        # 108. デキる猫は今日も憂鬱
        if 'デキる猫は今日も憂鬱' in title:
            title = re.sub(r'^特装版　', '', title)
            match = re.search(r'[０-９\d]+', title)
            if match:
                vol = match.group(0).translate(trans_table)
                return f'デキる猫は今日も憂鬱_original_{vol}'
            return 'デキる猫は今日も憂鬱_original_0'
        # 109. とんでもスキルで異世界放浪メシ
        if 'とんでもスキルで異世界放浪メシ' in title:
            title = re.sub(r'^特装版　', '', title)
            match = re.search(r'[０-９\d]+', title)
            if match:
                vol = match.group(0).translate(trans_table)
                return f'とんでもスキルで異世界放浪メシ_original_{vol}'
            return 'とんでもスキルで異世界放浪メシ_original_0'
        # 110. 生徒会にも穴はある！
        if '生徒会にも穴はある！' in title:
            title = re.sub(r'^特装版　', '', title)
            match = re.search(r'[０-９\d]+', title)
            if match:
                vol = match.group(0).translate(trans_table)
                return f'生徒会にも穴はある！_original_{vol}'
            return '生徒会にも穴はある！_original_0'
        # 111. 極楽街
        if '極楽街' in title:
            title = re.sub(r'^特装版　', '', title)
            match = re.search(r'[０-９\d]+', title)
            if match:
                vol = match.group(0).translate(trans_table)
                return f'極楽街_original_{vol}'
            return '極楽街_original_0'
        # 112. 夏目友人帳
        if '夏目友人帳' in title:
            title = re.sub(r'^特装版　', '', title)
            match = re.search(r'[０-９\d]+', title)
            if match:
                vol = match.group(0).translate(trans_table)
                return f'夏目友人帳_original_{vol}'
            return '夏目友人帳_original_0'
        # 113. ぐらんぶる
        if 'ぐらんぶる' in title:
            title = re.sub(r'^特装版　', '', title)
            match = re.search(r'[０-９\d]+', title)
            if match:
                vol = match.group(0).translate(trans_table)
                return f'ぐらんぶる_original_{vol}'
            return 'ぐらんぶる_original_0'
        # 114. 新装版　動物のお医者さん
        if '新装版　動物のお医者さん' in title:
            title = re.sub(r'^特装版　', '', title)
            match = re.search(r'[０-９\d]+', title)
            if match:
                vol = match.group(0).translate(trans_table)
                return f'新装版　動物のお医者さん_original_{vol}'
            return '新装版　動物のお医者さん_original_0'
        # 115. ふつうの軽音部
        if 'ふつうの軽音部' in title:
            title = re.sub(r'^特装版　', '', title)
            match = re.search(r'[０-９\d]+', title)
            if match:
                vol = match.group(0).translate(trans_table)
                return f'ふつうの軽音部_original_{vol}'
            return 'ふつうの軽音部_original_0'
        # 116. ＧＩＡＮＴ　ＫＩＬＬＩＮＧ
        if 'ＧＩＡＮＴ　ＫＩＬＬＩＮＧ' in title:
            title = re.sub(r'^特装版　', '', title)
            match = re.search(r'[０-９\d]+', title)
            if match:
                vol = match.group(0).translate(trans_table)
                return f'ＧＩＡＮＴ　ＫＩＬＬＩＮＧ_original_{vol}'
            return 'ＧＩＡＮＴ　ＫＩＬＬＩＮＧ_original_0'
        # 117. 四つ子ぐらし
        if '四つ子ぐらし' in title:
            title = re.sub(r'^特装版　', '', title)
            match = re.search(r'[０-９\d]+', title)
            if match:
                vol = match.group(0).translate(trans_table)
                return f'四つ子ぐらし_original_{vol}'
            return '四つ子ぐらし_original_0'
        # 118. 月が導く異世界道中
        if '月が導く異世界道中' in title:
            title = re.sub(r'^特装版　', '', title)
            match = re.search(r'[０-９\d]+', title)
            if match:
                vol = match.group(0).translate(trans_table)
                return f'月が導く異世界道中_original_{vol}'
            return '月が導く異世界道中_original_0'
        # 119. 墜落ＪＫと廃人教師
        if '墜落ＪＫと廃人教師' in title:
            title = re.sub(r'^特装版　', '', title)
            match = re.search(r'[０-９\d]+', title)
            if match:
                vol = match.group(0).translate(trans_table)
                return f'墜落ＪＫと廃人教師_original_{vol}'
            return '墜落ＪＫと廃人教師_original_0'
        # 120. きのう何食べた？
        if 'きのう何食べた？' in title:
            title = re.sub(r'^特装版　', '', title)
            match = re.search(r'[０-９\d]+', title)
            if match:
                vol = match.group(0).translate(trans_table)
                return f'きのう何食べた？_original_{vol}'
            return 'きのう何食べた？_original_0'
        # 121. Ｒｅ：ゼロから始める異世界生活
        if 'Ｒｅ：ゼロから始める異世界生活' in title:
            title = re.sub(r'^特装版　', '', title)
            match = re.search(r'[０-９\d]+', title)
            if match:
                vol = match.group(0).translate(trans_table)
                return f'Ｒｅ：ゼロから始める異世界生活_original_{vol}'
            return 'Ｒｅ：ゼロから始める異世界生活_original_0'
        # 122. ダークギャザリング
        if 'ダークギャザリング' in title:
            title = re.sub(r'^特装版　', '', title)
            match = re.search(r'[０-９\d]+', title)
            if match:
                vol = match.group(0).translate(trans_table)
                return f'ダークギャザリング_original_{vol}'
            return 'ダークギャザリング_original_0'
        # 123. 東京エイリアンズ
        if '東京エイリアンズ' in title:
            title = re.sub(r'^特装版　', '', title)
            match = re.search(r'[０-９\d]+', title)
            if match:
                vol = match.group(0).translate(trans_table)
                return f'東京エイリアンズ_original_{vol}'
            return '東京エイリアンズ_original_0'
        # 124. カッコウの許嫁
        if 'カッコウの許嫁' in title:
            title = re.sub(r'^特装版　', '', title)
            match = re.search(r'[０-９\d]+', title)
            if match:
                vol = match.group(0).translate(trans_table)
                return f'カッコウの許嫁_original_{vol}'
            return 'カッコウの許嫁_original_0'
        # 125. 君と宇宙を歩くために
        if '君と宇宙を歩くために' in title:
            title = re.sub(r'^特装版　', '', title)
            match = re.search(r'[０-９\d]+', title)
            if match:
                vol = match.group(0).translate(trans_table)
                return f'君と宇宙を歩くために_original_{vol}'
            return '君と宇宙を歩くために_original_0'
        # 126. 放課後ミステリクラブ
        if '放課後ミステリクラブ' in title:
            title = re.sub(r'^特装版　', '', title)
            match = re.search(r'[０-９\d]+', title)
            if match:
                vol = match.group(0).translate(trans_table)
                return f'放課後ミステリクラブ_original_{vol}'
            return '放課後ミステリクラブ_original_0'
        # 127. ザ・ファブル
        if 'ザ・ファブル' in title:
            title = re.sub(r'^特装版　', '', title)
            match = re.search(r'[０-９\d]+', title)
            if match:
                vol = match.group(0).translate(trans_table)
                return f'ザ・ファブル_original_{vol}'
            return 'ザ・ファブル_original_0'
        # 128. ようこそ実力至上主義の教室　２年生編
        if 'ようこそ実力至上主義の教室　２年生編' in title:
            title = re.sub(r'^特装版　', '', title)
            match = re.search(r'[０-９\d]+', title)
            if match:
                vol = match.group(0).translate(trans_table)
                return f'ようこそ実力至上主義の教室　２年生編_original_{vol}'
            return 'ようこそ実力至上主義の教室　２年生編_original_0'
        # 129. キン肉マン
        if 'キン肉マン' in title:
            title = re.sub(r'^特装版　', '', title)
            match = re.search(r'[０-９\d]+', title)
            if match:
                vol = match.group(0).translate(trans_table)
                return f'キン肉マン_original_{vol}'
            return 'キン肉マン_original_0'
        # 130. お隣の天使様にいつの間にか駄目人間に
        if 'お隣の天使様にいつの間にか駄目人間に' in title:
            title = re.sub(r'^特装版　', '', title)
            match = re.search(r'[０-９\d]+', title)
            if match:
                vol = match.group(0).translate(trans_table)
                return f'お隣の天使様にいつの間にか駄目人間に_original_{vol}'
            return 'お隣の天使様にいつの間にか駄目人間に_original_0'
        # 131. 転生したら第七王子だったので、気ま
        if '転生したら第七王子だったので、気ま' in title:
            title = re.sub(r'^特装版　', '', title)
            match = re.search(r'[０-９\d]+', title)
            if match:
                vol = match.group(0).translate(trans_table)
                return f'転生したら第七王子だったので、気ま_original_{vol}'
            return '転生したら第七王子だったので、気ま_original_0'
        # 132. 正反対な君と僕
        if '正反対な君と僕' in title:
            title = re.sub(r'^特装版　', '', title)
            match = re.search(r'[０-９\d]+', title)
            if match:
                vol = match.group(0).translate(trans_table)
                return f'正反対な君と僕_original_{vol}'
            return '正反対な君と僕_original_0'
        # 133. 日常ロック
        if '日常ロック' in title:
            title = re.sub(r'^特装版　', '', title)
            match = re.search(r'[０-９\d]+', title)
            if match:
                vol = match.group(0).translate(trans_table)
                return f'日常ロック_original_{vol}'
            return '日常ロック_original_0'
        # 134. メイドインアビス
        if 'メイドインアビス' in title:
            title = re.sub(r'^特装版　', '', title)
            match = re.search(r'[０-９\d]+', title)
            if match:
                vol = match.group(0).translate(trans_table)
                return f'メイドインアビス_original_{vol}'
            return 'メイドインアビス_original_0'
        # 135. ザ・ファブル　Ｔｈｅ　ｓｅｃｏｎｄ
        if 'ザ・ファブル　Ｔｈｅ　ｓｅｃｏｎｄ' in title:
            title = re.sub(r'^特装版　', '', title)
            match = re.search(r'[０-９\d]+', title)
            if match:
                vol = match.group(0).translate(trans_table)
                return f'ザ・ファブル　Ｔｈｅ　ｓｅｃｏｎｄ_original_{vol}'
            return 'ザ・ファブル　Ｔｈｅ　ｓｅｃｏｎｄ_original_0'
        # 136. 戦隊大失格
        if '戦隊大失格' in title:
            title = re.sub(r'^特装版　', '', title)
            match = re.search(r'[０-９\d]+', title)
            if match:
                vol = match.group(0).translate(trans_table)
                return f'戦隊大失格_original_{vol}'
            return '戦隊大失格_original_0'
        # 137. 仮面ライダーＤＶＤコレ平成　全国版
        if '仮面ライダーＤＶＤコレ平成　全国版' in title:
            title = re.sub(r'^特装版　', '', title)
            match = re.search(r'[０-９\d]+', title)
            if match:
                vol = match.group(0).translate(trans_table)
                return f'仮面ライダーＤＶＤコレ平成　全国版_original_{vol}'
            return '仮面ライダーＤＶＤコレ平成　全国版_original_0'
        # 138. 魔王城でおやすみ
        if '魔王城でおやすみ' in title:
            title = re.sub(r'^特装版　', '', title)
            match = re.search(r'[０-９\d]+', title)
            if match:
                vol = match.group(0).translate(trans_table)
                return f'魔王城でおやすみ_original_{vol}'
            return '魔王城でおやすみ_original_0'
        # 139. 金田一３７歳の事件簿
        if '金田一３７歳の事件簿' in title:
            title = re.sub(r'^特装版　', '', title)
            match = re.search(r'[０-９\d]+', title)
            if match:
                vol = match.group(0).translate(trans_table)
                return f'金田一３７歳の事件簿_original_{vol}'
            return '金田一３７歳の事件簿_original_0'
        # 140. シャドーハウス
        if 'シャドーハウス' in title:
            title = re.sub(r'^特装版　', '', title)
            match = re.search(r'[０-９\d]+', title)
            if match:
                vol = match.group(0).translate(trans_table)
                return f'シャドーハウス_original_{vol}'
            return 'シャドーハウス_original_0'
        # 141. 陰の実力者になりたくて！
        if '陰の実力者になりたくて！' in title:
            title = re.sub(r'^特装版　', '', title)
            match = re.search(r'[０-９\d]+', title)
            if match:
                vol = match.group(0).translate(trans_table)
                return f'陰の実力者になりたくて！_original_{vol}'
            return '陰の実力者になりたくて！_original_0'
        # 142. 空母いぶきＧＲＥＡＴ　ＧＡＭＥ
        if '空母いぶきＧＲＥＡＴ　ＧＡＭＥ' in title:
            title = re.sub(r'^特装版　', '', title)
            match = re.search(r'[０-９\d]+', title)
            if match:
                vol = match.group(0).translate(trans_table)
                return f'空母いぶきＧＲＥＡＴ　ＧＡＭＥ_original_{vol}'
            return '空母いぶきＧＲＥＡＴ　ＧＡＭＥ_original_0'
        # 143. ドッグスレッド
        if 'ドッグスレッド' in title:
            title = re.sub(r'^特装版　', '', title)
            match = re.search(r'[０-９\d]+', title)
            if match:
                vol = match.group(0).translate(trans_table)
                return f'ドッグスレッド_original_{vol}'
            return 'ドッグスレッド_original_0'
        # 144. ようこそ実力至上主義の教　２年生編
        if 'ようこそ実力至上主義の教　２年生編' in title:
            title = re.sub(r'^特装版　', '', title)
            match = re.search(r'[０-９\d]+', title)
            if match:
                vol = match.group(0).translate(trans_table)
                return f'ようこそ実力至上主義の教　２年生編_original_{vol}'
            return 'ようこそ実力至上主義の教　２年生編_original_0'
        # 145. ゴジラ＆東宝特撮ＯＦＦＩＣＩＡＬＭ
        if 'ゴジラ＆東宝特撮ＯＦＦＩＣＩＡＬＭ' in title:
            title = re.sub(r'^特装版　', '', title)
            match = re.search(r'[０-９\d]+', title)
            if match:
                vol = match.group(0).translate(trans_table)
                return f'ゴジラ＆東宝特撮ＯＦＦＩＣＩＡＬＭ_original_{vol}'
            return 'ゴジラ＆東宝特撮ＯＦＦＩＣＩＡＬＭ_original_0'
        # 146. 舞妓さんちのまかないさん
        if '舞妓さんちのまかないさん' in title:
            title = re.sub(r'^特装版　', '', title)
            match = re.search(r'[０-９\d]+', title)
            if match:
                vol = match.group(0).translate(trans_table)
                return f'舞妓さんちのまかないさん_original_{vol}'
            return '舞妓さんちのまかないさん_original_0'
        # 147. 無職転生～異世界行ったら本気だす～
        if '無職転生～異世界行ったら本気だす～' in title:
            title = re.sub(r'^特装版　', '', title)
            match = re.search(r'[０-９\d]+', title)
            if match:
                vol = match.group(0).translate(trans_table)
                return f'無職転生～異世界行ったら本気だす～_original_{vol}'
            return '無職転生～異世界行ったら本気だす～_original_0'
        # 148. あぶない刑事ＤＶＤコレクション全国版
        if 'あぶない刑事ＤＶＤコレクション全国版' in title:
            title = re.sub(r'^特装版　', '', title)
            match = re.search(r'[０-９\d]+', title)
            if match:
                vol = match.group(0).translate(trans_table)
                return f'あぶない刑事ＤＶＤコレクション全国版_original_{vol}'
            return 'あぶない刑事ＤＶＤコレクション全国版_original_0'
        # 149. 負けヒロインが多すぎる！
        if '負けヒロインが多すぎる！' in title:
            title = re.sub(r'^特装版　', '', title)
            match = re.search(r'[０-９\d]+', title)
            if match:
                vol = match.group(0).translate(trans_table)
                return f'負けヒロインが多すぎる！_original_{vol}'
            return '負けヒロインが多すぎる！_original_0'
        # 150. からかい上手の高木さん
        if 'からかい上手の高木さん' in title:
            title = re.sub(r'^特装版　', '', title)
            match = re.search(r'[０-９\d]+', title)
            if match:
                vol = match.group(0).translate(trans_table)
                return f'からかい上手の高木さん_original_{vol}'
            return 'からかい上手の高木さん_original_0'
        # 151. ブラッククローバー
        if 'ブラッククローバー' in title:
            title = re.sub(r'^特装版　', '', title)
            match = re.search(r'[０-９\d]+', title)
            if match:
                vol = match.group(0).translate(trans_table)
                return f'ブラッククローバー_original_{vol}'
            return 'ブラッククローバー_original_0'
        # 152. 花野井くんと恋の病
        if '花野井くんと恋の病' in title:
            title = re.sub(r'^特装版　', '', title)
            match = re.search(r'[０-９\d]+', title)
            if match:
                vol = match.group(0).translate(trans_table)
                return f'花野井くんと恋の病_original_{vol}'
            return '花野井くんと恋の病_original_0'
        # 153. アルスラーン戦記
        if 'アルスラーン戦記' in title:
            title = re.sub(r'^特装版　', '', title)
            match = re.search(r'[０-９\d]+', title)
            if match:
                vol = match.group(0).translate(trans_table)
                return f'アルスラーン戦記_original_{vol}'
            return 'アルスラーン戦記_original_0'
        # 154. からかい上手の（元）高木さん
        if 'からかい上手の（元）高木さん' in title:
            title = re.sub(r'^特装版　', '', title)
            match = re.search(r'[０-９\d]+', title)
            if match:
                vol = match.group(0).translate(trans_table)
                return f'からかい上手の（元）高木さん_original_{vol}'
            return 'からかい上手の（元）高木さん_original_0'
        # 155. 転生したら第七王子だったので、気まま
        if '転生したら第七王子だったので、気まま' in title:
            title = re.sub(r'^特装版　', '', title)
            match = re.search(r'[０-９\d]+', title)
            if match:
                vol = match.group(0).translate(trans_table)
                return f'転生したら第七王子だったので、気まま_original_{vol}'
            return '転生したら第七王子だったので、気まま_original_0'
        # 156. ＭＩＮＥＣＲＡＦＴ～世界の果てへの旅
        if 'ＭＩＮＥＣＲＡＦＴ～世界の果てへの旅' in title:
            title = re.sub(r'^特装版　', '', title)
            match = re.search(r'[０-９\d]+', title)
            if match:
                vol = match.group(0).translate(trans_table)
                return f'ＭＩＮＥＣＲＡＦＴ～世界の果てへの旅_original_{vol}'
            return 'ＭＩＮＥＣＲＡＦＴ～世界の果てへの旅_original_0'
        # 157. 転生賢者の異世界ライフ～第二の職業
        if '転生賢者の異世界ライフ～第二の職業' in title:
            title = re.sub(r'^特装版　', '', title)
            match = re.search(r'[０-９\d]+', title)
            if match:
                vol = match.group(0).translate(trans_table)
                return f'転生賢者の異世界ライフ～第二の職業_original_{vol}'
            return '転生賢者の異世界ライフ～第二の職業_original_0'
        # 158. 来世は他人がいい
        if '来世は他人がいい' in title:
            title = re.sub(r'^特装版　', '', title)
            match = re.search(r'[０-９\d]+', title)
            if match:
                vol = match.group(0).translate(trans_table)
                return f'来世は他人がいい_original_{vol}'
            return '来世は他人がいい_original_0'
        # 159. 鵺の陰陽師
        if '鵺の陰陽師' in title:
            title = re.sub(r'^特装版　', '', title)
            match = re.search(r'[０-９\d]+', title)
            if match:
                vol = match.group(0).translate(trans_table)
                return f'鵺の陰陽師_original_{vol}'
            return '鵺の陰陽師_original_0'
        # 160. 女神のカフェテラス
        if '女神のカフェテラス' in title:
            title = re.sub(r'^特装版　', '', title)
            match = re.search(r'[０-９\d]+', title)
            if match:
                vol = match.group(0).translate(trans_table)
                return f'女神のカフェテラス_original_{vol}'
            return '女神のカフェテラス_original_0'
        # 161. 風都探偵
        if '風都探偵' in title:
            title = re.sub(r'^特装版　', '', title)
            match = re.search(r'[０-９\d]+', title)
            if match:
                vol = match.group(0).translate(trans_table)
                return f'風都探偵_original_{vol}'
            return '風都探偵_original_0'
        # 162. 鬼滅の刃
        if '鬼滅の刃' in title:
            # 外伝
            if '外伝' in title:
                return '鬼滅の刃_外伝_0'
            # ノベライズシリーズ
            if 'ノベライズ' in title:
                return '鬼滅の刃_ノベライズ_0'
            # 小説シリーズ
            if any(x in title for x in ['風の道しるべ', 'しあわせの花', '片羽の蝶']):
                return '鬼滅の刃_小説_0'
            # 塗絵帳シリーズ
            if '塗絵帳' in title:
                return '鬼滅の刃_塗絵帳_0'
            # ファンブック
            if 'ファンブック' in title or '鬼殺隊見聞録' in title:
                return '鬼滅の刃_ファンブック_0'
            # キャラクターズブック
            if 'キャラクターズ' in title:
                return '鬼滅の刃_キャラクターズ_0'
            # イラスト記録集
            if 'イラスト記録集' in title:
                return '鬼滅の刃_イラスト記録集_0'
            # 画集
            if '画集' in title:
                return '鬼滅の刃_画集_0'
            # 劇場版
            if '劇場版' in title:
                return '鬼滅の刃_劇場版_0'
            # 雑誌特別版
            if 'メンズノンノ' in title:
                return '鬼滅の刃_雑誌特別版_0'
            # アクアビーズなど商品
            if any(x in title for x in ['アクアビーズ', 'ヒノカミ血風譚']):
                return '鬼滅の刃_関連商品_0'
            # 特装版（本編）
            if '特装版' in title or 'グッズ付き' in title:
                match = re.search(r'[０-９\d]+', title)
                if match:
                    vol = match.group(0).translate(trans_table)
                    return f'鬼滅の刃_特装版_{vol}'
                return '鬼滅の刃_特装版_0'
            # その他関連書籍
            if any(x in title for x in ['で哲学', 'で学ぶ', 'で心理分析']):
                return '鬼滅の刃_関連書籍_0'
            # 本編
            match = re.search(r'鬼滅の刃　+([０-９\d]+)$', title)
            if match:
                vol = match.group(1).translate(trans_table)
                return f'鬼滅の刃_original_{vol}'
            return '鬼滅の刃_original_0'
        # 週刊誌・月刊誌は正規化不要（そのまま返す）
        weekly_magazines = [
            'ＮＨＫラジオ', '週刊少年ジャンプ', '週刊文春', '週刊女性セブン', 'ＮＨＫきょうの料理',
            'ａｎａｎ', '週刊女性自身', 'コロコロコミック', '週刊新潮', '週刊少年マガジン',
            '週刊ポスト', '週刊ダイヤモンド', 'プレジデント', '週刊少年サンデー', '週刊ＴＶガイド',
            '文藝春秋', '週刊現代', 'ヤングジャンプ', 'ＢＲＵＴＵＳ', 'クロワッサン',
            'りぼん', 'めばえ', 'ＡＥＲＡ', 'ＦＲＩＤＡＹ', 'ちゃお'
        ]
        for magazine in weekly_magazines:
            if magazine in title:
                return original_title
        return original_title

    df['書名'] = df['書名'].apply(process_title)
    return df


def remove_volume_number(df, remove_series=False):
    """
    書名から「_巻数」部分を除去する
    前提: normalize_title()で「作品_シリーズ_巻数」形式に統一されていること

    引数:
        df: DataFrame
        remove_series: Trueの場合、シリーズ名も除去する

    例:
        remove_series=False:
            「作品_シリーズ_巻数」→「作品_シリーズ」
            「作品_巻数」→「作品」
            「作品」→「作品」
        remove_series=True:
            「作品_シリーズ_巻数」→「作品」
            「作品_巻数」→「作品」
            「作品」→「作品」
    """
    def process_volume(title):
        if pd.isna(title):
            return title
        if '_' not in title:
            return title
        parts = title.split('_')
        if remove_series:
            return parts[0]
        else:
            if len(parts) >= 3:
                return f"{parts[0]}_{parts[1]}"
            else:
                return parts[0]
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


def fill_publisher_by_ISBN(df):
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
    return df

def merge_store_detail(df, store_detail):
    df = df.merge(store_detail, on='書店コード', how='left')
    return df


def clean_df(df, store_detail):
    df = df.drop('Unnamed: 0', axis=1)
    df = df.dropna(subset=['出版社', '書名', '著者名', '本体価格'], how='all').copy()

    df = clean_time(df)
    df = fill_publisher_by_ISBN(df)
    df = normalize_author(df)
    df = normalize_title(df, remove_series=True)

    delete_space_columns = df.select_dtypes(include=['object']).columns.tolist()
    df = fill_missing_class(df)
    df = merge_store_detail(df, store_detail)
    df = delete_space(df, delete_space_columns)
    #df = remove_volume_number(df)

    return df

"""
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
"""
