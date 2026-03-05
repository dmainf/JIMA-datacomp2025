import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

# ==============================
# 設定
# ==============================

MODELS = [
    {
        "dir": "chronos_predict+GateRAF(base)/chronos_bolt(DoRA)+RAF",
        "csv": "prediction_difference_all.csv",
        "label": "Gate+RAF"
    },
    
    {
        "dir": "sarima/sarima_baseline",
        "csv": "prediction_difference_all.csv",
        "label": "SARIMA"
    },
    
    #{
    #    "dir": "chronos_predict+ProtoRAF(base)/chronos_bolt(DoRA)",
    #    "csv": "prediction_difference_all.csv",
    #    "label": "DoRA"
    #},
    #{
    #    "dir": "chronos_predict+ProtoRAF(base)/chronos_bolt(zero-shot)",
    #    "csv": "prediction_difference_all.csv",
    #    "label": "zero-shot"
    #}
]

OUTPUT_FILE = "chronos_SARIMA_boxplot.png"

# ==============================
# データ読み込み
# ==============================

data_list = []
labels = []

for model in MODELS:

    path = os.path.join(model["dir"], model["csv"])

    df = pd.read_csv(path)

    # 条件フィルタ
    df = df[df["Total_Sales"] >= 64]

    # Difference_100を取得
    values = df["Difference_100"].dropna()

    data_list.append(values)
    labels.append(model["label"])

    print(f"{model['label']} : {len(values)} samples")

# ==============================
# 箱ひげ図
# ==============================

plt.figure(figsize=(10,6))

plt.boxplot(
    data_list,
    labels=labels,
    whis=[0,100],   # ← 最小値～最大値
    showfliers=False
)

# 赤い破線（100）
plt.axhline(
    y=100,
    color="red",
    linestyle="--",
    linewidth=2
)


plt.ylabel("Difference_100")
plt.title("Prediction Difference (TotalSales >= 64)")
plt.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig(OUTPUT_FILE, dpi=300)

print("Saved :", OUTPUT_FILE)