import os
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CONFIG = {
    "chronos": {
        "spike_dir": "../chronos_raf(spike)/chronos_bolt(DoRA)+RAF",
        "spike_csv": "evaluation_all_spike.csv",
        "diff_dir": "../chronos_predict+GateRAF(base)/chronos_bolt(DoRA)+RAF",
        "diff_csv": "prediction_difference_all.csv",
        "label": "Gate+RAF",
    },
    "sarima": {
        "spike_dir": "../chronos_raf(spike)/chronos_bolt(DoRA)+RAF",
        "spike_csv": "evaluation_all_spike.csv",
        "diff_dir": "../sarima/sarima_baseline",
        "diff_csv": "prediction_difference_all.csv",
        "label": "SARIMA",
    },

    "min_n_spike": 10,
    "min_total_sales": 64,

    "output_png": "boxplot_nspike_10.png",
    "title": "Difference_100 (books with n_spike=>10 & Total_Sales>=64)",
    "y_label": "Difference_100",
    "whisker_minmax": True,
    "draw_red_100": True,
}

BOOK_COL_CANDIDATES = ["book_name", "書名", "Book", "title", "name"]
NSPIKE_COL_CANDIDATES = ["n_spike", "N_spike", "nSpike", "spike_count", "Spike_Count"]
TOTAL_COL_CANDIDATES = ["Total_Sales", "TotalSales", "total_sales", "TOTAL_SALES", "合計販売冊数", "総販売冊数"]
DIFF_COL_CANDIDATES = ["Difference_100", "difference_100", "Diff_100", "diff100"]

FORCE_BOOK_COL = None
FORCE_NSPIKE_COL = None
FORCE_TOTAL_COL = None
FORCE_DIFF_COL = None

def detect_col(df: pd.DataFrame, candidates, force=None, kind="column"):
    if force is not None:
        if force not in df.columns:
            raise ValueError(f"Not Found FORCE_{kind}='{force}' List = {list(df.columns)}")
        return force

    for c in candidates:
        if c in df.columns:
            return c

    lower_map = {col.lower(): col for col in df.columns}
    for c in candidates:
        lc = c.lower()
        if lc in lower_map:
            return lower_map[lc]

    raise ValueError(f"Not Found : {kind}, List = {list(df.columns)}")

def load_spike_bookset(spike_path: Path, min_n_spike: int):
    df = pd.read_csv(spike_path)

    book_col = detect_col(df, BOOK_COL_CANDIDATES, force=FORCE_BOOK_COL, kind="BOOK_COL")
    nspike_col = detect_col(df, NSPIKE_COL_CANDIDATES, force=FORCE_NSPIKE_COL, kind="NSPIKE_COL")

    df[nspike_col] = pd.to_numeric(df[nspike_col], errors="coerce")
    books = set(df.loc[df[nspike_col] >= min_n_spike, book_col].dropna().astype(str).unique())
    #books = set(df.loc[df[nspike_col] == 0, book_col].dropna().astype(str).unique())

    return books, book_col, nspike_col, len(df)


def load_diff_values(diff_path: Path, book_set: set, min_total_sales: int):
    df = pd.read_csv(diff_path)

    book_col = detect_col(df, BOOK_COL_CANDIDATES, force=FORCE_BOOK_COL, kind="BOOK_COL")
    total_col = detect_col(df, TOTAL_COL_CANDIDATES, force=FORCE_TOTAL_COL, kind="TOTAL_COL")
    diff_col = detect_col(df, DIFF_COL_CANDIDATES, force=FORCE_DIFF_COL, kind="DIFF_COL")

    df[book_col] = df[book_col].astype(str)
    df[total_col] = pd.to_numeric(df[total_col], errors="coerce")
    df[diff_col] = pd.to_numeric(df[diff_col], errors="coerce")

    df_f = df[df[book_col].isin(book_set)]
    df_f = df_f[df_f[total_col] >= min_total_sales]
    vals = df_f[diff_col].dropna().values

    return vals, (book_col, total_col, diff_col), len(df), len(df_f)

def main():
    min_n_spike = CONFIG["min_n_spike"]
    min_total_sales = CONFIG["min_total_sales"]

    results = []

    for key in ["chronos", "sarima"]:
        spec = CONFIG[key]
        label = spec["label"]

        spike_path = Path(spec["spike_dir"]) / spec["spike_csv"]
        diff_path  = Path(spec["diff_dir"])  / spec["diff_csv"]

        if not spike_path.exists():
            raise FileNotFoundError(f"[{label}] Not Found spike CSV : {spike_path}")
        if not diff_path.exists():
            raise FileNotFoundError(f"[{label}] Not Found Difference CSV : {diff_path}")

        book_set, bcol_s, ncol, n_spike_rows = load_spike_bookset(spike_path, min_n_spike)
        vals, (bcol_d, tcol, dcol), n_diff_rows, n_after = load_diff_values(diff_path, book_set, min_total_sales)

        print(f"\n=== {label} ===")
        print(f"spike_csv : {spike_path} (rows={n_spike_rows})")
        print(f"  used cols: book='{bcol_s}', n_spike='{ncol}'")
        print(f"  extracted books (n_spike>={min_n_spike}): {len(book_set)}")

        print(f"diff_csv  : {diff_path} (rows={n_diff_rows})")
        print(f"  used cols: book='{bcol_d}', total='{tcol}', diff='{dcol}'")
        print(f"  after filter (book in set & total>={min_total_sales}): rows={n_after}")
        print(f"  values count (non-NaN Difference_100): {len(vals)}")

        results.append((label, vals))

    labels = [r[0] for r in results]
    data = [r[1] for r in results]

    plt.figure(figsize=(8, 6))

    boxplot_kwargs = dict(labels=labels, showfliers=False)
    if CONFIG["whisker_minmax"]:
        boxplot_kwargs["whis"] = [0, 100]

    plt.boxplot(data, **boxplot_kwargs)

    if CONFIG["draw_red_100"]:
        plt.axhline(100, color="red", linestyle="--", linewidth=2)

    plt.title(CONFIG["title"])
    plt.ylabel(CONFIG["y_label"])
    plt.grid(True, axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(CONFIG["output_png"], dpi=300)

    print(f"\nSaved: {CONFIG['output_png']}")

if __name__ == "__main__":
    main()