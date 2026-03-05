import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

chronos_spike_csv = "../chronos_raf(spike)/chronos_bolt(DoRA)+RAF/evaluation_all_spike.csv"
chronos_diff_csv  = "../chronos_predict+GateRAF(base)/chronos_bolt(DoRA)+RAF/prediction_difference_all.csv"

sarima_spike_csv  = "../chronos_raf(spike)/chronos_bolt(DoRA)+RAF/evaluation_all_spike.csv"
sarima_diff_csv   = "../sarima/sarima_baseline/prediction_difference_all.csv"

OUTLIER_UPPER = 1800
MIN_TOTAL_SALES = 64

OUT_BOTH   = "spike_vs_difference_both_no_outlier.png"
OUT_CHRON  = "spike_vs_difference_chronos_no_outlier.png"
OUT_SARIMA = "spike_vs_difference_sarima_no_outlier.png"

TITLE_BASE = f"Spike Frequency vs Prediction Ratio (Total_Sales>={MIN_TOTAL_SALES}, Diff< {OUTLIER_UPPER})"
DRAW_RED_100 = True

def load_and_merge(spike_csv, diff_csv, label):
    if not os.path.exists(spike_csv):
        raise FileNotFoundError(f"[{label}] spike CSV not found: {spike_csv}")
    if not os.path.exists(diff_csv):
        raise FileNotFoundError(f"[{label}] diff CSV not found: {diff_csv}")

    spike_df = pd.read_csv(spike_csv)
    diff_df  = pd.read_csv(diff_csv)

    if "n_spike" in spike_df.columns:
        spike_df["n_spike"] = pd.to_numeric(spike_df["n_spike"], errors="coerce")
    diff_df["Difference_100"] = pd.to_numeric(diff_df["Difference_100"], errors="coerce")
    diff_df["Total_Sales"] = pd.to_numeric(diff_df["Total_Sales"], errors="coerce")

    merged = pd.merge(
        spike_df[["book_name", "n_spike"]],
        diff_df[["book_name", "Difference_100", "Total_Sales"]],
        on="book_name",
        how="inner"
    )

    merged = merged[merged["Total_Sales"] >= MIN_TOTAL_SALES]
    merged = merged[merged["Difference_100"] < OUTLIER_UPPER]

    merged = merged.dropna(subset=["n_spike", "Difference_100"])

    return merged

def setup_axes(ax, title):
    ax.set_title(title)
    ax.set_xlabel("Number of Spikes (n_spike)")
    ax.set_ylabel("Difference_100")
    ax.grid(True)
    if DRAW_RED_100:
        ax.axhline(100, color="red", linestyle="--", linewidth=2)

def save_scatter(df, label, out_path, title):
    fig, ax = plt.subplots(figsize=(10, 6))

    if label == "Gate+RAF":
        color = "tab:blue"
    elif label == "SARIMA":
        color = "tab:orange"
    else:
        color = "black"

    ax.scatter(
        df["n_spike"],
        df["Difference_100"],
        alpha=0.6,
        label=label,
        color=color
    )

    setup_axes(ax, title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

chronos = load_and_merge(chronos_spike_csv, chronos_diff_csv, "Gate+RAF")
sarima  = load_and_merge(sarima_spike_csv,  sarima_diff_csv,  "SARIMA")

print("=== After filters ===")
print(f"Gate+RAF: {len(chronos)} points")
print(f"SARIMA  : {len(sarima)} points")

fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(chronos["n_spike"], chronos["Difference_100"], alpha=0.6, label="Gate+RAF", color="tab:blue")
ax.scatter(sarima["n_spike"],  sarima["Difference_100"],  alpha=0.6, label="SARIMA", color="tab:orange")
setup_axes(ax, TITLE_BASE + " (Both)")
ax.legend()
fig.tight_layout()
fig.savefig(OUT_BOTH, dpi=300)
plt.close(fig)

save_scatter(
    chronos,
    label="Gate+RAF",
    out_path=OUT_CHRON,
    title=TITLE_BASE + " (Gate+RAF)"
)

save_scatter(
    sarima,
    label="SARIMA",
    out_path=OUT_SARIMA,
    title=TITLE_BASE + " (SARIMA)"
)

print("\nSaved:")
print(" ", OUT_BOTH)
print(" ", OUT_CHRON)
print(" ", OUT_SARIMA)