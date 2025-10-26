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

    # 全角数字を半角に変換するテーブル
    trans_table = str.maketrans('０１２３４５６７８９', '0123456789')

    def process_title(title):
        if pd.isna(title):
            return title

        original_title = title

        # ========== 作品別の個別処理 ========== #

        # 1. 薬屋のひとりごと
        if title.startswith('薬屋のひとりごと'):
            # 特装版を除去
            title = re.sub(r'^特装版　', '', title)
            # 限定特装版を除去
            title = re.sub(r'　+限定特装版$', '', title)
            # バリューパック等を除去
            title = re.sub(r'　+バリューパック$', '', title)
            # 画集は別物として扱う
            if '画集' in title:
                return '薬屋のひとりごと_画集'
            # サブシリーズ：猫猫の後宮謎解き手
            if '～猫猫の後宮謎解き手' in title:
                match = re.search(r'～猫猫の後宮謎解き手　+([０-９\d]+)', title)
                if match:
                    vol = match.group(1).translate(trans_table)
                    return f'薬屋のひとりごと_猫猫の後宮謎解き手{vol}'
            # サブシリーズ：猫猫の後宮謎解き（「手」なし）
            if '～猫猫の後宮謎解き' in title or '～猫猫の後' in title:
                match = re.search(r'～猫猫の後[宮謎解き]*　+([０-９\d]+)', title)
                if match:
                    vol = match.group(1).translate(trans_table)
                    return f'薬屋のひとりごと_猫猫の後宮謎解き{vol}'
            # 通常の巻数
            match = re.search(r'薬屋のひとりごと　+([０-９\d]+)', title)
            if match:
                vol = match.group(1).translate(trans_table)
                return f'薬屋のひとりごと_{vol}'
            return '薬屋のひとりごと'

        # 2. 呪術廻戦
        if title.startswith('呪術廻戦') or '呪術廻戦' in title:
            # 関連書籍（本編以外）
            if any(x in title for x in ['で英語を学ぶ', '公式ファンブック', '塗絵帳', 'ＴＶアニメ', '１２０％楽しむ']):
                # 関連書籍は統合
                if 'ＴＶアニメ' in title:
                    return '呪術廻戦_TVアニメ関連'
                elif '公式ファンブック' in title:
                    return '呪術廻戦_公式ファンブック'
                elif '塗絵帳' in title:
                    return '呪術廻戦_塗絵帳'
                else:
                    return '呪術廻戦_関連書籍'
            # 劇場版
            if title.startswith('劇場版'):
                if 'ノベライズ' in title:
                    return '呪術廻戦_劇場版0ノベライズ'
                return '呪術廻戦_劇場版0'
            # 小説
            if '夜明けのいばら道' in title or '逝く夏と還る秋' in title:
                return '呪術廻戦_小説'
            # 本編の巻数
            title = re.sub(r'　+同梱版$|　+カレンダー同梱版$', '', title)
            match = re.search(r'呪術廻戦　+([０-９\d]+)', title)
            if match:
                vol = match.group(1).translate(trans_table)
                return f'呪術廻戦_{vol}'
            return '呪術廻戦'

        # 3. ブルーロック
        if 'ブルーロック' in title:
            # アニメ関連
            if 'アニメブルーロック' in title or 'ＴＶアニメブルーロック' in title:
                return 'ブルーロック_アニメ関連'
            # キャラクターブック
            if 'キャラクターブック' in title:
                return 'ブルーロック_キャラクターブック'
            # 関連書籍
            if any(x in title for x in ['公式ぬりえ', '生存　漢字ドリル', '４７都道府県']):
                return 'ブルーロック_関連書籍'
            # 小説
            if title.startswith('小説'):
                match = re.search(r'[　\s]+([０-９\d]+)', title)
                if match:
                    vol = match.group(1).translate(trans_table)
                    if 'ＥＰＩＳＯＤＥ凪' in title:
                        return f'ブルーロック_小説EPISODE凪{vol}'
                    return f'ブルーロック_小説{vol}'
                return 'ブルーロック_小説'
            # EPISODE凪シリーズ
            if 'ＥＰＩＳＯＤＥ　凪' in title or 'ＥＰＩＳＯＤＥ凪' in title:
                match = re.search(r'[　\s]+([０-９\d]+)', title)
                if match:
                    vol = match.group(1).translate(trans_table)
                    return f'ブルーロック_EPISODE凪{vol}'
            # 特装版を除去
            title = re.sub(r'　+特装版$', '', title)
            # 本編
            match = re.search(r'ブルーロック　+([０-９\d]+)', title)
            if match:
                vol = match.group(1).translate(trans_table)
                return f'ブルーロック_{vol}'
            return 'ブルーロック'

        # 4. 葬送のフリーレン
        if '葬送のフリーレン' in title:
            # 関連書籍
            if any(x in title for x in ['ポスターコレクション', 'アンソロジー', 'クリアファイル', 'プレミアムポストカード',
                                        '公式ファンブック', '画集', 'ＴＶアニメ', '日めくり', 'コミック付箋']):
                return '葬送のフリーレン_関連書籍'
            # 小説
            if title.startswith('小説') or '魂の眠る地への旅路' in title:
                return '葬送のフリーレン_小説'
            # 特装版を除去
            title = re.sub(r'　+特装版$', '', title)
            # 本編
            match = re.search(r'葬送のフリーレン　+([０-９\d]+)', title)
            if match:
                vol = match.group(1).translate(trans_table)
                return f'葬送のフリーレン_{vol}'
            return '葬送のフリーレン'

        # 5. ハイキュー
        if 'ハイキュー' in title:
            # スピンオフ：れっつ！ハイキュー！？
            if 'れっつ！ハイキュー！？' in title:
                match = re.search(r'[　\s]+([０-９\d]+)', title)
                if match:
                    vol = match.group(1).translate(trans_table)
                    return f'ハイキュー_れっつ{vol}'
            # スピンオフ：ハイキュー部！！
            if 'ハイキュー部！！' in title:
                match = re.search(r'[　\s]+([０-９\d]+)', title)
                if match:
                    vol = match.group(1).translate(trans_table)
                    return f'ハイキュー_ハイキュー部{vol}'
            # 小説版
            if 'ショーセツバン' in title:
                match = re.search(r'[　\s]+([０-９\d]+)', title)
                if match:
                    vol = match.group(1).translate(trans_table)
                    return f'ハイキュー_小説{vol}'
                return 'ハイキュー_小説'
            # 関連書籍
            if any(x in title for x in ['ファイナルガイド', 'クロニクル', 'セイシュンメイカン', 'カラーイラスト',
                                        'Ｍａｇａｚｉｎ', 'ＴＶアニメチームブック', '劇場版', '排球本', '生き方がラク',
                                        'カレンダー', 'Ｃｏｍｐｌｅｔｅ']):
                return 'ハイキュー_関連書籍'
            # 本編
            match = re.search(r'ハイキュー！！　+([０-９\d]+)', title)
            if match:
                vol = match.group(1).translate(trans_table)
                return f'ハイキュー_{vol}'
            return 'ハイキュー'

        # 6. ＯＮＥＰＩＥＣＥ
        if 'ＯＮＥＰＩＥＣＥ' in title:
            match = re.search(r'[　\s]+([０-９\d]+)', title)
            if match:
                vol = match.group(1).translate(trans_table)
                return f'ＯＮＥＰＩＥＣＥ_{vol}'
            return 'ＯＮＥＰＩＥＣＥ'

        # 7. 怪獣８号
        if '怪獣８号' in title:
            # スピンオフ
            if 'ＲＥＬＡＸ' in title:
                match = re.search(r'[　\s]+([０-９\d]+)', title)
                if match:
                    vol = match.group(1).translate(trans_table)
                    return f'怪獣８号_RELAX{vol}'
            if 'ｓｉｄｅ　Ｂ' in title:
                match = re.search(r'[　\s]+([０-９\d]+)', title)
                if match:
                    vol = match.group(1).translate(trans_table)
                    return f'怪獣８号_sideB{vol}'
            # 関連書籍
            if any(x in title for x in ['密着', '日本防衛隊', '終焉のフォルティチュード']):
                return '怪獣８号_関連書籍'
            # 本編
            match = re.search(r'怪獣８号　+([０-９\d]+)', title)
            if match:
                vol = match.group(1).translate(trans_table)
                return f'怪獣８号_{vol}'
            return '怪獣８号'

        # 8. 僕のヒーローアカデミア
        if '僕のヒーローアカデミア' in title:
            # スピンオフ：ヴィジランテ
            if 'ヴィジランテ' in title:
                match = re.search(r'[　\s]+([０-９\d]+)', title)
                if match:
                    vol = match.group(1).translate(trans_table)
                    return f'僕のヒーローアカデミア_ヴィジランテ{vol}'
            # スピンオフ：すまっしゅ
            if 'すまっしゅ' in title:
                match = re.search(r'[　\s]+([０-９\d]+)', title)
                if match:
                    vol = match.group(1).translate(trans_table)
                    return f'僕のヒーローアカデミア_すまっしゅ{vol}'
            # スピンオフ：チームアップ
            if 'チームアップ' in title:
                match = re.search(r'[　\s]+([０-９\d]+)', title)
                if match:
                    vol = match.group(1).translate(trans_table)
                    return f'僕のヒーローアカデミア_チームアップ{vol}'
            # スピンオフ：雄英白書
            if '雄英白書' in title:
                match = re.search(r'[　\s]+([０-９\d]+)', title)
                if match:
                    vol = match.group(1).translate(trans_table)
                    return f'僕のヒーローアカデミア_雄英白書{vol}'
                return '僕のヒーローアカデミア_雄英白書'
            # 関連書籍
            if any(x in title for x in ['かんたんイラスト', '公式キャラクター', 'ＴＨＥ', 'ＴＶアニメ', 'カレンダー']):
                return '僕のヒーローアカデミア_関連書籍'
            # 本編
            match = re.search(r'僕のヒーローアカデミア　+([０-９\d]+)', title)
            if match:
                vol = match.group(1).translate(trans_table)
                return f'僕のヒーローアカデミア_{vol}'
            return '僕のヒーローアカデミア'

        # 9. キングダム
        if 'キングダム' in title and not any(x in title for x in ['ツキウタ', 'オレ様', 'アニア', '恐竜', '餃子', '迷宮', 'サキュバス']):
            # 完全版
            if '完全版' in title:
                match = re.search(r'[　\s]+([０-９\d]+)', title)
                if match:
                    vol = match.group(1).translate(trans_table)
                    return f'キングダム_完全版{vol}'
            # 映画
            if '映画' in title or '劇場版' in title or '大将軍の帰還' in title or '運命の炎' in title:
                return 'キングダム_映画'
            # 関連書籍
            if any(x in title for x in ['公式ガイド', '公式問題集', '英雄風雲録', '水晶玉子', 'ビジュアル']):
                return 'キングダム_関連書籍'
            # 本編
            match = re.search(r'キングダム　+([０-９\d]+)', title)
            if match:
                vol = match.group(1).translate(trans_table)
                return f'キングダム_{vol}'
            return 'キングダム'

        # 10. ダンダダン
        if 'ダンダダン' in title:
            # 関連書籍
            if '超常現象解体新書' in title:
                return 'ダンダダン_関連書籍'
            # 本編
            match = re.search(r'ダンダダン　+([０-９\d]+)', title)
            if match:
                vol = match.group(1).translate(trans_table)
                return f'ダンダダン_{vol}'
            return 'ダンダダン'

        # 11. ＷＩＮＤ　ＢＲＥＡＫＥＲ
        if 'ＷＩＮＤ　ＢＲＥＡＫＥＲ' in title:
            # 関連書籍
            if '公式キャラクタ' in title:
                return 'ＷＩＮＤ　ＢＲＥＡＫＥＲ_関連書籍'
            # 本編
            match = re.search(r'[　\s]+([０-９\d]+)', title)
            if match:
                vol = match.group(1).translate(trans_table)
                return f'ＷＩＮＤ　ＢＲＥＡＫＥＲ_{vol}'
            return 'ＷＩＮＤ　ＢＲＥＡＫＥＲ'

        # 12. 【推しの子】
        if '【推しの子】' in title:
            # スピンオフ
            if '一番星のスピカ' in title:
                return '【推しの子】_一番星のスピカ'
            if '二人のエチュード' in title:
                return '【推しの子】_二人のエチュード'
            # 関連書籍
            if any(x in title for x in ['まんがノベライズ', 'カラーリング', 'イラスト集', '公式ガイ', '映画', 'セット', 'ＴＶアニメ', 'カレンダー']):
                return '【推しの子】_関連書籍'
            # 特装版を除去
            title = re.sub(r'　+ＳＰＥＣＩＡＬ.*$', '', title)
            # 本編
            match = re.search(r'【推しの子】　+([０-９\d]+)', title)
            if match:
                vol = match.group(1).translate(trans_table)
                return f'【推しの子】_{vol}'
            return '【推しの子】'

        # 13. アオのハコ
        if 'アオのハコ' in title:
            # Prologue
            if 'Ｐｒｏｌｏｇｕｅ' in title:
                return 'アオのハコ_Prologue'
            # 本編
            match = re.search(r'アオのハコ　+([０-９\d]+)', title)
            if match:
                vol = match.group(1).translate(trans_table)
                return f'アオのハコ_{vol}'
            return 'アオのハコ'

        # 14. ＮＨＫラジオラジオ英会話
        if 'ＮＨＫラジオラジオ英会話' in title:
            return 'ＮＨＫラジオラジオ英会話'

        # 15. 変な家
        if title.startswith('変な家'):
            if '文庫版' in title:
                return '変な家_文庫版'
            match = re.search(r'変な家　+([０-９\d]+)', title)
            if match:
                vol = match.group(1).translate(trans_table)
                return f'変な家_{vol}'
            return '変な家'

        # 16. 週刊少年ジャンプ
        if '週刊少年ジャンプ' in title:
            return '週刊少年ジャンプ'

        # 17. 転生したらスライムだった件
        if '転生したらスライムだった件' in title:
            # スピンオフ：転ちゅら
            if '転ちゅら' in title:
                match = re.search(r'[　\s]+([０-９\d]+)', title)
                if match:
                    vol = match.group(1).translate(trans_table)
                    return f'転生したらスライムだった件_転ちゅら{vol}'
            # スピンオフ：クレイマ
            if 'クレイマ' in title:
                match = re.search(r'[　\s]+([０-９\d]+)', title)
                if match:
                    vol = match.group(1).translate(trans_table)
                    return f'転生したらスライムだった件_クレイマ{vol}'
            # スピンオフ：異聞～魔国
            if '異聞～魔国' in title:
                match = re.search(r'[　\s]+([０-９\d]+)', title)
                if match:
                    vol = match.group(1).translate(trans_table)
                    return f'転生したらスライムだった件_異聞魔国{vol}'
            # スピンオフ：～魔物の国
            if '～魔物の国' in title:
                match = re.search(r'[　\s]+([０-９\d]+)', title)
                if match:
                    vol = match.group(1).translate(trans_table)
                    return f'転生したらスライムだった件_魔物の国{vol}'
            # スピンオフ：美食伝
            if '美食伝' in title:
                match = re.search(r'[　\s]+([０-９\d]+)', title)
                if match:
                    vol = match.group(1).translate(trans_table)
                    return f'転生したらスライムだった件_美食伝{vol}'
            # スピンオフ：番外編
            if '番外編' in title:
                match = re.search(r'[　\s]+([０-９\d]+)', title)
                if match:
                    vol = match.group(1).translate(trans_table)
                    return f'転生したらスライムだった件_番外編{vol}'
            # 関連書籍
            if any(x in title for x in ['異世界サバイ', '転生漢字ドリ', 'で学べる', '公式キャラクタ', 'ＡＮＩＭＥ']):
                return '転生したらスライムだった件_関連書籍'
            # 特装版等を除去
            title = re.sub(r'　+(限定版|特装版|バリューパック)$', '', title)
            # 上中下巻
            match = re.search(r'転生したらスライムだった件　+([０-９\d\.]+)　+(上|中|下)', title)
            if match:
                vol = match.group(1).translate(trans_table)
                part = match.group(2)
                return f'転生したらスライムだった件_{vol}{part}'
            # 全３巻セット
            if '全３巻' in title:
                match = re.search(r'[　\s]+([０-９\d]+)', title)
                if match:
                    vol = match.group(1).translate(trans_table)
                    return f'転生したらスライムだった件_{vol}'
            # 通常巻数
            match = re.search(r'転生したらスライムだった件　+([０-９\d\.]+)', title)
            if match:
                vol = match.group(1).translate(trans_table)
                return f'転生したらスライムだった件_{vol}'
            return '転生したらスライムだった件'

        # 18. 魔入りました
        if '魔入りました' in title:
            # 小説
            if title.startswith('小説'):
                match = re.search(r'[　\s]+([０-９\d]+)', title)
                if match:
                    vol = match.group(1).translate(trans_table)
                    return f'魔入りました_小説{vol}'
            # スピンオフ
            if 'ｉｆ　Ｅｐｉ' in title:
                match = re.search(r'[　\s]+([０-９\d]+)', title)
                if match:
                    vol = match.group(1).translate(trans_table)
                    return f'魔入りました_ifEpi{vol}'
            if '外伝' in title:
                match = re.search(r'[　\s]+([０-９\d]+)', title)
                if match:
                    vol = match.group(1).translate(trans_table)
                    return f'魔入りました_外伝{vol}'
            # 関連書籍
            if any(x in title for x in ['スターターＢＯＸ', '公式アンソロ', '公式ファンブック', 'で学ぶ']):
                return '魔入りました_関連書籍'
            # 本編
            match = re.search(r'魔入りました！入間くん　+([０-９\d]+)', title)
            if match:
                vol = match.group(1).translate(trans_table)
                return f'魔入りました_{vol}'
            return '魔入りました'

        # 19. ＳＰＹ×ＦＡＭＩＬＹ
        if 'ＳＰＹ×ＦＡＭＩＬＹ' in title:
            # 関連書籍
            if any(x in title for x in ['まんがノベライ', 'アニメーション', 'オペレーション', '家族の肖像',
                                        'フォージャー家の謎', '公式ファンブック', 'ＴＶアニメ', 'カレンダー', '劇場版']):
                return 'ＳＰＹ×ＦＡＭＩＬＹ_関連書籍'
            # 特装版・同梱版を除去
            title = re.sub(r'　+(特装版|同梱版)$', '', title)
            # 本編
            match = re.search(r'ＳＰＹ×ＦＡＭＩＬＹ　+([０-９\d]+)', title)
            if match:
                vol = match.group(1).translate(trans_table)
                return f'ＳＰＹ×ＦＡＭＩＬＹ_{vol}'
            return 'ＳＰＹ×ＦＡＭＩＬＹ'

        # 20. ダンジョン飯
        if 'ダンジョン飯' in title:
            # 関連書籍
            if any(x in title for x in ['ワールドガイド', '冒険者バイ', '英会話', 'Ｗａｌｋｅｒ']):
                return 'ダンジョン飯_関連書籍'
            # 本編
            match = re.search(r'ダンジョン飯　+([０-９\d]+)', title)
            if match:
                vol = match.group(1).translate(trans_table)
                return f'ダンジョン飯_{vol}'
            return 'ダンジョン飯'

        # 21. 名探偵コナン（非常に多様なため、簡略化）
        if '名探偵コナン' in title:
            # 本編
            match = re.search(r'名探偵コナン　+([０-９\d]+)', title)
            if match:
                vol = match.group(1).translate(trans_table)
                # 特装版を除去
                return f'名探偵コナン_{vol}'
            # その他は全て統合
            return '名探偵コナン_関連'

        # 22. マッシュル
        if 'マッシュル' in title:
            match = re.search(r'[　\s]+([０-９\d]+)', title)
            if match:
                vol = match.group(1).translate(trans_table)
                return f'マッシュル_{vol}'
            return 'マッシュル'

        # 23. ＳＡＫＡＭＯＴＯ　ＤＡＹＳ
        if 'ＳＡＫＡＭＯＴＯ　ＤＡＹＳ' in title:
            # 関連書籍
            if any(x in title for x in ['殺し屋のメソ', '殺し屋ブルー']):
                return 'ＳＡＫＡＭＯＴＯ　ＤＡＹＳ_関連書籍'
            match = re.search(r'[　\s]+([０-９\d]+)', title)
            if match:
                vol = match.group(1).translate(trans_table)
                return f'ＳＡＫＡＭＯＴＯ　ＤＡＹＳ_{vol}'
            return 'ＳＡＫＡＭＯＴＯ　ＤＡＹＳ'

        # 24. ＨＵＮＴＥＲ×ＨＵＮＴＥＲ
        if 'ＨＵＮＴＥＲ×ＨＵＮＴＥＲ' in title:
            match = re.search(r'[　\s]+([０-９\d]+)', title)
            if match:
                vol = match.group(1).translate(trans_table)
                return f'ＨＵＮＴＥＲ×ＨＵＮＴＥＲ_{vol}'
            return 'ＨＵＮＴＥＲ×ＨＵＮＴＥＲ'

        # 25. チェンソーマン
        if 'チェンソーマン' in title:
            # 関連書籍
            if any(x in title for x in ['バディ・ストーリーズ', 'ＴＶアニメ']):
                return 'チェンソーマン_関連書籍'
            match = re.search(r'チェンソーマン　+([０-９\d]+)', title)
            if match:
                vol = match.group(1).translate(trans_table)
                return f'チェンソーマン_{vol}'
            return 'チェンソーマン'

        # 26. 週刊文春
        if '週刊文春' in title:
            return '週刊文春'

        # 27. 逃げ上手の若君
        if '逃げ上手の若君' in title:
            match = re.search(r'逃げ上手の若君　+([０-９\d]+)', title)
            if match:
                vol = match.group(1).translate(trans_table)
                return f'逃げ上手の若君_{vol}'
            return '逃げ上手の若君'

        # 28. 週刊女性セブン
        if '週刊女性セブン' in title:
            return '週刊女性セブン'

        # 29. 地縛少年花子くん
        if '地縛少年花子くん' in title:
            # 画集
            if '画集' in title:
                match = re.search(r'[　\s]+([０-９\d]+)', title)
                if match:
                    vol = match.group(1).translate(trans_table)
                    return f'地縛少年花子くん_画集{vol}'
                return '地縛少年花子くん_画集'
            match = re.search(r'地縛少年花子くん　+([０-９\d]+)', title)
            if match:
                vol = match.group(1).translate(trans_table)
                return f'地縛少年花子くん_{vol}'
            return '地縛少年花子くん'

        # 30. 夜桜さんちの大作戦
        if '夜桜さんちの大作戦' in title:
            # 関連書籍
            if any(x in title for x in ['おるすばん', '観察日記']):
                return '夜桜さんちの大作戦_関連書籍'
            match = re.search(r'夜桜さんちの大作戦　+([０-９\d]+)', title)
            if match:
                vol = match.group(1).translate(trans_table)
                return f'夜桜さんちの大作戦_{vol}'
            return '夜桜さんちの大作戦'

        # 31. ＮＨＫきょうの料理
        if 'ＮＨＫきょうの料理' in title:
            return 'ＮＨＫきょうの料理'

        # 32. ａｎａｎ
        if title.startswith('ａｎａｎ'):
            return 'ａｎａｎ'

        # 33. 週刊女性自身
        if '週刊女性自身' in title:
            return '週刊女性自身'

        # 34. アオアシ
        if 'アオアシ' in title:
            # ジュニア版
            if 'ジュニア版' in title:
                match = re.search(r'[　\s]+([０-９\d]+)', title)
                if match:
                    vol = match.group(1).translate(trans_table)
                    return f'アオアシ_ジュニア版{vol}'
            # スピンオフ：ブラザーフット
            if 'ブラザーフット' in title:
                match = re.search(r'[　\s]+([０-９\d]+)', title)
                if match:
                    vol = match.group(1).translate(trans_table)
                    return f'アオアシ_ブラザーフット{vol}'
            # 小説
            if title.startswith('小説'):
                match = re.search(r'[　\s]+([０-９\d]+)', title)
                if match:
                    vol = match.group(1).translate(trans_table)
                    return f'アオアシ_小説{vol}'
            # 関連書籍
            if 'に学ぶ' in title:
                return 'アオアシ_関連書籍'
            # 本編
            match = re.search(r'アオアシ　+([０-９\d]+)', title)
            if match:
                vol = match.group(1).translate(trans_table)
                return f'アオアシ_{vol}'
            return 'アオアシ'

        # 35. コロコロコミック
        if 'コロコロコミック' in title:
            return 'コロコロコミック'

        # 36. つかめ！理科ダマン
        if 'つかめ！理科ダマン' in title:
            match = re.search(r'[　\s]+([０-９\d]+)', title)
            if match:
                vol = match.group(1).translate(trans_table)
                return f'つかめ！理科ダマン_{vol}'
            return 'つかめ！理科ダマン'

        # 37. 週刊新潮
        if '週刊新潮' in title:
            return '週刊新潮'

        # 38. ＮＨＫラジオ英会話タイムトライアル
        if 'ＮＨＫラジオ英会話タイムトライアル' in title:
            return 'ＮＨＫラジオ英会話タイムトライアル'

        # 39. ゴールデンカムイ
        if 'ゴールデンカムイ' in title:
            # 関連書籍
            if any(x in title for x in ['アイヌ文化', '絵から学ぶ', '公式フ', '映画']):
                return 'ゴールデンカムイ_関連書籍'
            # 本編
            match = re.search(r'ゴールデンカムイ　+([０-９\d]+)', title)
            if match:
                vol = match.group(1).translate(trans_table)
                return f'ゴールデンカムイ_{vol}'
            return 'ゴールデンカムイ'

        # 40. 忘却バッテリー
        if '忘却バッテリー' in title:
            match = re.search(r'忘却バッテリー　+([０-９\d]+)', title)
            if match:
                vol = match.group(1).translate(trans_table)
                return f'忘却バッテリー_{vol}'
            return '忘却バッテリー'

        # 41. シャングリラ・フロンティア
        if 'シャングリラ・フロンティア' in title:
            # 関連書籍
            if 'るるぶ' in title:
                return 'シャングリラ・フロンティア_関連書籍'
            # 特装版・限定版を除去
            title = re.sub(r'　+(特装版|限定版)$', '', title)
            # クソゲーハンター版
            if '～クソゲ' in title:
                match = re.search(r'[　\s]+([０-９\d]+)', title)
                if match:
                    vol = match.group(1).translate(trans_table)
                    return f'シャングリラ・フロンティア_クソゲーハンター{vol}'
            # 通常版
            match = re.search(r'シャングリラ・フロンティア　+([０-９\d]+)', title)
            if match:
                vol = match.group(1).translate(trans_table)
                return f'シャングリラ・フロンティア_{vol}'
            return 'シャングリラ・フロンティア'

        # 42. 週刊少年マガジン
        if '週刊少年マガジン' in title:
            return '週刊少年マガジン'

        # 43. 時々ボソッとロシア語でデレる隣のアー
        if '時々ボソッとロシア語でデレる隣のアー' in title:
            match = re.search(r'[　\s]+([０-９\d]+)', title)
            if match:
                vol = match.group(1).translate(trans_table)
                return f'時々ボソッとロシア語でデレる隣のアーリャさん_{vol}'
            return '時々ボソッとロシア語でデレる隣のアーリャさん'

        # 44. カグラバチ
        if 'カグラバチ' in title:
            match = re.search(r'カグラバチ　+([０-９\d]+)', title)
            if match:
                vol = match.group(1).translate(trans_table)
                return f'カグラバチ_{vol}'
            return 'カグラバチ'

        # 45. 大ピンチずかん
        if '大ピンチずかん' in title:
            match = re.search(r'[　\s]+([０-９\d]+)', title)
            if match:
                vol = match.group(1).translate(trans_table)
                return f'大ピンチずかん_{vol}'
            return '大ピンチずかん'

        # 46. 文藝春秋
        if '文藝春秋' in title:
            return '文藝春秋'

        # 47. 週刊ＴＶガイド
        if '週刊ＴＶガイド' in title:
            return '週刊ＴＶガイド'

        # 48. 僕の心のヤバイやつ
        if '僕の心のヤバイやつ' in title:
            # 特装版を除去
            title = re.sub(r'^特装版　', '', title)
            # 小説
            if title.startswith('小説'):
                return '僕の心のヤバイやつ_小説'
            # 関連書籍
            if 'ＴＶアニメ' in title:
                return '僕の心のヤバイやつ_関連書籍'
            match = re.search(r'僕の心のヤバイやつ　+([０-９\d]+)', title)
            if match:
                vol = match.group(1).translate(trans_table)
                return f'僕の心のヤバイやつ_{vol}'
            return '僕の心のヤバイやつ'

        # 49. 週刊ポスト
        if '週刊ポスト' in title:
            return '週刊ポスト'

        # 50. 週刊ダイヤモンド
        if '週刊ダイヤモンド' in title:
            return '週刊ダイヤモンド'

        # ========== その他の一般的なパターン ========== #

        # 一般的なパターン：「タイトル　巻数」
        match = re.match(r'^(.+?)　+([０-９\d]+)$', title)
        if match:
            base_title = match.group(1)
            volume = match.group(2).translate(trans_table)
            return f"{base_title}_{volume}"

        # 変更がなければ元のタイトルを返す
        return original_title

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
