import numpy as np
import pandas as pd
from numba import jit
from typing import Tuple


@jit(nopython=True)
def compute_regime_features(
    sales: np.ndarray,
    window_size: int = 5,
    hawkes_decay: float = 0.1,
    initial_adi: float = 30.0,
    initial_cv2: float = 1.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(sales)

    feat_adi = np.full(n, initial_adi, dtype=np.float32)
    feat_cv2 = np.full(n, initial_cv2, dtype=np.float32)
    feat_hawkes = np.zeros(n, dtype=np.float32)
    feat_days_since = np.full(n, np.inf, dtype=np.float32)

    current_adi = initial_adi
    current_cv2 = initial_cv2
    last_hawkes = 0.0
    last_event_day = -np.inf

    interval_buffer = np.zeros(window_size, dtype=np.float32)
    buffer_count = 0
    buffer_idx = 0

    for t in range(n):
        feat_days_since[t] = t - last_event_day if last_event_day > -np.inf else np.inf

        feat_adi[t] = current_adi
        feat_cv2[t] = current_cv2

        dt = 1.0
        current_hawkes_val = last_hawkes * np.exp(-hawkes_decay * dt)
        feat_hawkes[t] = current_hawkes_val

        if sales[t] > 0:
            last_hawkes = current_hawkes_val + sales[t]

            if last_event_day > -np.inf:
                new_interval = t - last_event_day

                interval_buffer[buffer_idx] = new_interval
                buffer_idx = (buffer_idx + 1) % window_size
                buffer_count = min(buffer_count + 1, window_size)

                if buffer_count > 0:
                    valid_intervals = interval_buffer[:buffer_count] if buffer_count < window_size else interval_buffer
                    mean_interval = np.mean(valid_intervals)
                    std_interval = np.std(valid_intervals)

                    current_adi = mean_interval
                    if mean_interval > 0:
                        current_cv2 = (std_interval / mean_interval) ** 2
                    else:
                        current_cv2 = 1.0

            last_event_day = float(t)
        else:
            last_hawkes = current_hawkes_val

    return feat_adi, feat_cv2, feat_hawkes, feat_days_since


def add_regime_features(
    df: pd.DataFrame,
    item_col: str = '書名',
    date_col: str = '日付',
    sales_col: str = 'POS販売冊数',
    window_size: int = 5,
    hawkes_decay: float = 0.1,
    initial_adi: float = 30.0,
    initial_cv2: float = 1.0
) -> pd.DataFrame:
    df = df.sort_values([item_col, date_col]).reset_index(drop=True)

    results = []

    for item_name, group in df.groupby(item_col, observed=True):
        sales_array = group[sales_col].values.astype(np.float32)

        feat_adi, feat_cv2, feat_hawkes, feat_days_since = compute_regime_features(
            sales_array,
            window_size=window_size,
            hawkes_decay=hawkes_decay,
            initial_adi=initial_adi,
            initial_cv2=initial_cv2
        )

        results.append(pd.DataFrame({
            'feat_adi': feat_adi,
            'feat_cv2': feat_cv2,
            'feat_hawkes': feat_hawkes,
            'feat_days_since': feat_days_since
        }, index=group.index))

    features_df = pd.concat(results, axis=0).sort_index()

    result_df = pd.concat([df, features_df], axis=1)

    return result_df


def add_regime_scores(
    df: pd.DataFrame,
    lambda_adi: float = 30.0,
    scale_hawkes: float = 10.0
) -> pd.DataFrame:
    df = df.copy()

    df['score_sparse'] = 1.0 - np.exp(-df['feat_adi'] / lambda_adi)

    df['score_periodic'] = np.clip(1.0 - np.sqrt(df['feat_cv2']), 0.0, 1.0)

    df['score_burst'] = np.tanh(df['feat_hawkes'] / scale_hawkes)

    return df


def plot_regime_patterns(
    df: pd.DataFrame,
    n_quantiles: int = 11,
    item_col: str = '書名',
    date_col: str = '日付',
    sales_col: str = 'POS販売冊数',
    output_dir: str = 'regime_plots'
):
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib import rcParams
    import os

    rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Hiragino Sans', 'Yu Gothic', 'Meiryo']
    rcParams['axes.unicode_minus'] = False

    os.makedirs(output_dir, exist_ok=True)

    total_sales = df.groupby(item_col, observed=True)[sales_col].sum().sort_values(ascending=False)

    percentiles = np.linspace(0, 100, n_quantiles)
    selected_items = []
    for p in percentiles:
        target_sales = np.percentile(total_sales.values, 100 - p)
        closest_item = (total_sales - target_sales).abs().idxmin()
        if closest_item not in selected_items:
            selected_items.append(closest_item)

    selected_items = selected_items[:n_quantiles]

    for idx, item_name in enumerate(selected_items):
        item_data = df[df[item_col] == item_name].sort_values(date_col)

        dates = pd.to_datetime(item_data[date_col])
        sales = item_data[sales_col].values
        score_sparse = item_data['score_sparse'].values
        score_periodic = item_data['score_periodic'].values
        score_burst = item_data['score_burst'].values

        total_sale = sales.sum()
        percentile_rank = (total_sales > total_sale).sum() / len(total_sales) * 100

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))

        ax1.fill_between(dates, 0, sales, alpha=0.3, color='steelblue')
        ax1.plot(dates, sales, color='steelblue', linewidth=1.5, label='POS販売冊数 (log1p)')
        ax1.set_ylabel('販売冊数 (log1p)', fontsize=12)
        ax1.set_title(f'{item_name}\n総売上: {total_sale:,.0f}冊 (上位{percentile_rank:.1f}%)', fontsize=13, fontweight='bold')
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper right', fontsize=10)

        ax2.plot(dates, score_sparse, label='Sparse (過疎)', color='gray', linewidth=2, alpha=0.8)
        ax2.plot(dates, score_periodic, label='Periodic (周期)', color='green', linewidth=2, alpha=0.8)
        ax2.plot(dates, score_burst, label='Burst (バースト)', color='red', linewidth=2, alpha=0.8)
        ax2.set_ylabel('パターンスコア', fontsize=12)
        ax2.set_xlabel('日付', fontsize=12)
        ax2.set_ylim(-0.05, 1.05)
        ax2.set_title('需要パターン分類スコア (0~1)', fontsize=13, fontweight='bold')
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper right', fontsize=10)
        ax2.axhline(y=0.5, color='black', linestyle='--', linewidth=0.8, alpha=0.3)

        for ax in [ax1, ax2]:
            ax.tick_params(axis='x', rotation=45)

        plt.tight_layout()

        safe_filename = f"{idx+1:02d}_{item_name[:30].replace('/', '_')}.png"
        output_path = os.path.join(output_dir, safe_filename)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  {idx+1}/11: {safe_filename}")
        plt.close()

    print(f"\n全{len(selected_items)}ファイルを {output_dir}/ に保存しました")


if __name__ == '__main__':
    df = pd.read_parquet('df_for.parquet')
    df['POS販売冊数'] = np.log1p(df['POS販売冊数'])

    print("元データ:")
    print(df.head())
    print(f"\nデータサイズ: {len(df):,} 行")
    print(f"書名数: {df['書名'].nunique():,} 件")

    print("\n特徴量生成中...")
    df_with_features = add_regime_features(df)

    print("\n特徴量の統計:")
    print(df_with_features[['feat_adi', 'feat_cv2', 'feat_hawkes', 'feat_days_since']].describe())

    print("\nスコア計算中...")
    df_scored = add_regime_scores(df_with_features)

    print("\nスコアの統計:")
    print(df_scored[['score_sparse', 'score_periodic', 'score_burst']].describe())

    output_path = 'df_with_regime_scores.parquet'
    df_scored.to_parquet(output_path, index=False)
    print(f"\n保存完了: {output_path}")

    print("\n需要パターンのプロット作成中...")
    plot_regime_patterns(df_scored, n_quantiles=11, output_dir='regime_plots')
    print("\n完了")
