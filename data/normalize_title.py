import pandas as pd
from mlx_lm import load, generate
from tqdm import tqdm

model_id = "mlx-community/Qwen2.5-7B-Instruct-4bit"
THRESHHOLD = 6880 #before normalize tutle, 出現回数 >=100

print("=== Loading Data ===")
df_raw = pd.read_csv('title_list.csv')
print("=== Complete! ===")

print(f"=== Loading Model ===")
model, tokenizer = load(model_id)
print("=== Complete! ===")

system_prompt = (
    "あなたは書誌情報の正規化スペシャリストだ．"
    "入力された `書名` を分析し，以下のルールに従って `作品名_シリーズ名_巻数` の形式に変換せよ．"
    "出力は正規化後の文字列のみとし，余計な会話は一切禁止する．\n\n"
    "【正規化ルール】\n"
    "1. **フォーマット**: `作品名_シリーズ名_巻数` とする（区切りは半角アンダースコア）．\n"
    "2. **基本**: シリーズ名がない本編は `original` とする．巻数がない場合は `0`．\n"
    "3. **ノイズ除去**: 全角スペースは削除．「特装版」「増刊」「ワイド」などの装飾語は無視する．\n"
    "4. **シリーズ判定**: 「ＥＰＩＳＯＤＥ凪」や「猫猫の後宮謎解き」のような副題はシリーズ名として抽出する．\n"
    "5. **雑誌・年鑑**: 「2024年春号」や「'25」などの時系列情報は，巻数またはシリーズ名として適切に保持する．\n\n"
    "6. **上下巻**: 「白鳥とコウモリ　下」などの上下巻情報は，上は1巻，下は2巻として処理する\n\n"
    "【Few-Shot Examples (実データに基づく模範解答)】\n"
    "入力: 呪術廻戦　２５\n出力: 呪術廻戦_original_25\n"
    "入力: 怪獣８号　　　８\n出力: 怪獣８号_original_8\n"
    "入力: ブルーロック－ＥＰＩＳＯＤＥ　凪－　４\n出力: ブルーロック_EPISODE凪_4\n"
    "入力: 薬屋のひとりごと～猫猫の後宮謎解き　１８\n出力: 薬屋のひとりごと_猫猫の後宮謎解き手_18\n"
    "入力: 会社四季報増　２０２４年３集夏号ワイド\n出力: 会社四季報_季節号_2024夏\n"
    "入力: '２５　会社四季報　業界地図\n出力: 会社四季報_業界地図_2025\n"
    "入力: ＮＨＫラジオラジオ英会話\n出力: ＮＨＫラジオ_ラジオ英会話_0\n"
    "入力: 週　刊　文　春\n出力: 週刊文春_original_0\n"
    "入力: 首都圏版月刊ザ・テレビジョン\n出力: 月刊ザ・テレビジョン_首都圏版\n"
    "【以前あなたが間違えた正規化とその正解】\n"
    "入力: 呪術廻戦　２２\n間違え：咒术回战_original_22\n出力:呪術廻戦_original_22 \n"
    "入力: ＢＲＵＴＵＳ（ブルータス）\n間違え：ブルータス_original_0\n出力:ＢＲＵＴＵＳ_original_0 \n"
    "入力: ＳｐｏｒｔｓＧｒａｐｈｉｃ　Ｎｕｍｂｅｒ\n間違え：ＳｐｏｒｔｓＧｒａｐｈｉｃ_Ｎｕｍｂｅｒ\n出力: ＳｐｏｒｔｓＧｒａｐｈｉｃＮｕｍｂｅｒ_original_0\n"
    "入力: ＴＶ　ｆａｎ\n間違え：TV_fan_0\n出力: TVfan_original_0\n"
    "入力: ＯＮＥ　ＰＩＥＣＥ　１０８\n間違え：ONE PIECE_original_108\n出力: ONEPIECE_original_108\n"
    "入力: 白鳥とコウモリ　下\n間違え：白鳥とコウモリ_original_1\n出力: 白鳥とコウモリ_original_2\n"
    "入力: 幼　稚　園\n間違え：original_0\n出力: 幼稚園_original_0\n"
    "入力: ベ　ス　ト　カ　ー\n間違え：best_car_original_0\n出力: ベストカー_original_0\n"
    "入力: 首都圏版月刊ザ・テレビジョン\n間違え：月刊ザ・テレビジョン_首都圏版\n出力: 月刊ザ・テレビジョン_首都圏版_0\n"
    "入力: キングダム　７１\n間違え：kingダム_original_71\n出力: キングダム_original_71\n"
)

def create_prompt(raw_title):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"入力: {str(raw_title)}\n出力:"},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

titles = df_raw['書名'].astype(str).head(THRESHHOLD).tolist()
normalized_results = []

print(f"Processing {len(titles)} records with MLX...")
for title in tqdm(titles):
    prompt = create_prompt(title)

    output = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=64,
        verbose=False
    )

    normalized_results.append(output.strip())

df_raw["normalized_title"] = pd.Series(normalized_results, index=df_raw.head(THRESHHOLD).index)

print("=== Saving Results ===")
df_raw.head(len(normalized_results)).to_csv('normalized_title_list.csv', index=False)