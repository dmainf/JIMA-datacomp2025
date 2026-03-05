from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DEFAULT_EVAL_DIR = "sarima_baseline"

def _save_boxplot(values: np.ndarray, out_png: Path, ylabel: str, title: str) -> None:
    if values.size == 0:
        raise ValueError(f"No valid values to plot: {out_png.name}")

    plt.figure(figsize=(6, 6))
    plt.boxplot(values, vert=True, whis=[0, 100], showfliers=False)
    plt.axhline(y=100, color="red", linestyle="--", linewidth=1.5)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir",
        dest="eval_dir",
        default=DEFAULT_EVAL_DIR,
        help="Directory that contains evaluation_all.csv (default is set in code).",
    )
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    in_path = eval_dir / "evaluation_all.csv"
    out_csv = eval_dir / "prediction_difference_all.csv"

    out_png_1 = eval_dir / "boxplot_difference_counts.png"
    out_png_2 = eval_dir / "boxplot_difference_100_all.png"
    out_png_3 = eval_dir / "boxplot_difference_100_total_ge_64.png"

    if not in_path.exists():
        raise FileNotFoundError(f"Not found: {in_path}")

    df = pd.read_csv(in_path, encoding="utf-8-sig")

    required = ["book_name", "Total_Sales", "Prediction_Total_Sales"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"evaluation_all.csv is missing required columns: {missing}")

    total = pd.to_numeric(df["Total_Sales"], errors="coerce")
    pred_total = pd.to_numeric(df["Prediction_Total_Sales"], errors="coerce")

    diff = pred_total - total

    with np.errstate(divide="ignore", invalid="ignore"):
        diff_100 = (pred_total / total) * 100.0
    diff_100 = diff_100.where(total != 0, np.nan)

    out = pd.DataFrame(
        {
            "book_name": df["book_name"].astype(str),
            "Total_Sales": total,
            "Prediction_Total_Sales": pred_total,
            "Difference": diff,
            "Difference_100": diff_100,
        }
    )

    out.to_csv(out_csv, index=False, encoding="utf-8-sig")

    _save_boxplot(
        out["Difference"].dropna().values,
        out_png_1,
        ylabel="Difference (冊数) = Prediction_Total_Sales - Total_Sales",
        title="Difference (counts) distribution",
    )

    _save_boxplot(
        out["Difference_100"].dropna().values,
        out_png_2,
        ylabel="Difference_100 (Total_Sales = 100)",
        title="Prediction/Actual (×100) distribution (ALL)",
    )

    filtered = out[(out["Total_Sales"] >= 64) & (out["Difference_100"].notna())]
    _save_boxplot(
        filtered["Difference_100"].values,
        out_png_3,
        ylabel="Difference_100 (Total_Sales = 100)",
        title="Prediction/Actual (×100) distribution (Total_Sales >= 64)",
    )

    print(f"Saved: {out_csv}")
    print(f"Saved: {out_png_1}")
    print(f"Saved: {out_png_2}")
    print(f"Saved: {out_png_3}")

if __name__ == "__main__":
    main()
