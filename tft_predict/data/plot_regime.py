import numpy as np
import pandas as pd
from numba import jit
from typing import Tuple
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import rcParams
import os

@jit(nopython=True, cache=True)
def find_first_significant_peak(
    rho: np.ndarray,
    min_lag: int,
    max_lag: int,
    threshold_ratio: float = 0.75,
    min_score: float = 0.25
) -> Tuple[int, float]:
    """
    改良版First Significant Peak探索（Red Noise対策付き）

    1. Strict Local Peak: ρ(τ-1) < ρ(τ) > ρ(τ+1) を必須条件化
       → AR(1)プロセスの単調減衰を除外
    2. 最小スコア閾値: min_score未満の弱い相関を無視
       → ノイズによる誤検知を防止
    3. First Peak優先: 倍音よりも基本波を優先

    Args:
        rho: 自己相関ベクトル
        min_lag: 最小ラグ
        max_lag: 最大ラグ
        threshold_ratio: 最大値に対する閾値比率（0.75推奨）
        min_score: 最小相関スコア（0.25推奨、これ未満は周期なしとする）

    Returns:
        (best_lag, max_corr): 検出された周期と相関値
    """
    global_max = -1.0
    for tau in range(min_lag, max_lag + 1):
        if rho[tau] > global_max:
            global_max = rho[tau]

    if global_max < min_score:
        return 0, 0.0

    threshold = max(global_max * threshold_ratio, min_score)

    for tau in range(min_lag, max_lag):
        val = rho[tau]

        if val >= threshold:
            if val > rho[tau - 1] and val > rho[tau + 1]:
                return tau, global_max

    return 0, 0.0


@jit(nopython=True, cache=True)
def compute_online_periodicity(
    sales: np.ndarray,
    min_lag: int = 3,
    max_lag: int = 60,
    alpha: float = 0.05,
    threshold_ratio: float = 0.75,
    min_score: float = 0.25
) -> Tuple[np.ndarray, np.ndarray]:
    """
    オンライン自己相関 + Red Noise対策付きピーク選択による周期性スコア計算

    改良点：
    - Strict Local Peak: AR(1)プロセスの単調減衰を除外
    - 最小スコア閾値: 弱い相関による誤検知を防止
    - First Significant Peak: 倍音よりも基本波を優先

    Args:
        sales: 売上時系列（対数変換済み推奨）
        min_lag: 最小周期（日、3推奨で2日ノイズを除外）
        max_lag: 最大周期（日）
        alpha: 学習率（0.01-0.1推奨、大きいほど変化に敏感）
        threshold_ratio: ピーク検出閾値（0.75推奨、大きいほど厳格）
        min_score: 最小相関スコア（0.25推奨、これ未満は周期なし）

    Returns:
        scores: 周期性スコア（0-1、高いほど周期的）
        detected_periods: 検出された周期（日数、0は周期なし）
    """
    n = len(sales)
    scores = np.zeros(n, dtype=np.float32)
    detected_periods = np.zeros(n, dtype=np.float32)

    mu = sales[0]
    var = 0.0

    rho = np.zeros(max_lag + 2, dtype=np.float32)
    history = np.zeros(max_lag + 1, dtype=np.float32)

    for t in range(n):
        val = sales[t]

        diff = val - mu
        mu = mu + alpha * diff
        var = (1.0 - alpha) * var + alpha * (diff * (val - mu))

        std = 1.0 if var < 1e-6 else np.sqrt(var)

        z_t = (val - mu) / std
        z_t = max(-5.0, min(5.0, z_t))

        for lag in range(max_lag, 0, -1):
            history[lag] = history[lag - 1]
        history[0] = z_t

        for lag in range(max_lag + 1):
            rho[lag] = (1.0 - alpha) * rho[lag] + alpha * (z_t * history[lag])

        if t >= min_lag:
            best_lag, max_rho = find_first_significant_peak(
                rho, min_lag, max_lag, threshold_ratio, min_score
            )

            scores[t] = max_rho
            detected_periods[t] = float(best_lag)

    return scores, detected_periods


@jit(nopython=True, cache=True)
def compute_regime_features(
    sales: np.ndarray,
    hawkes_decay: float = 0.1,
    initial_adi: float = 30.0,
    initial_cv2: float = 1.0,
    base_alpha: float = 0.1,
    min_period_lag: int = 3,
    max_period_lag: int = 60,
    period_alpha: float = 0.05,
    period_threshold_ratio: float = 0.75,
    period_min_score: float = 0.25
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    オンライン自己相関 + Red Noise対策による周期性検出を含む特徴量計算

    Args:
        min_period_lag: 最小周期（デフォルト3で2日ノイズを除外）
        period_threshold_ratio: 倍音抑制の閾値比率
            - 0.75 (推奨): バランス型、7日周期vs14日周期で基本波を優先
            - 0.85 (厳格): より強い倍音抑制
            - 0.65 (寛容): 弱い倍音抑制
        period_min_score: 最小相関スコア（0.25推奨）
            - これ未満の弱い相関は周期なしとする
            - AR(1)ノイズによる誤検知を防止
    """
    n = len(sales)

    feat_adi = np.full(n, initial_adi, dtype=np.float32)
    feat_cv2 = np.full(n, initial_cv2, dtype=np.float32)
    feat_hawkes = np.zeros(n, dtype=np.float32)
    feat_days_since = np.full(n, np.inf, dtype=np.float32)

    current_adi = initial_adi
    current_cv2 = initial_cv2
    current_sales_mean_interval = initial_adi
    current_sales_var_interval = 0.0

    last_hawkes = 0.0
    last_event_day = -1.0 - initial_adi

    for t in range(n):
        days_since_sales = float(t) - last_event_day
        feat_days_since[t] = days_since_sales

        effective_adi = max(current_adi, days_since_sales)
        feat_adi[t] = effective_adi
        feat_cv2[t] = current_cv2

        dt = 1.0
        current_hawkes_val = last_hawkes * np.exp(-hawkes_decay * dt)
        feat_hawkes[t] = current_hawkes_val

        if sales[t] > 0:
            last_hawkes = current_hawkes_val + sales[t]
            if last_event_day >= 0:
                new_interval = t - last_event_day
                delta = new_interval - current_sales_mean_interval
                current_sales_mean_interval += base_alpha * delta
                current_sales_var_interval = (1.0 - base_alpha) * (current_sales_var_interval + base_alpha * delta**2)

                current_adi = current_sales_mean_interval
                if current_sales_mean_interval > 1e-6:
                    current_cv2 = (np.sqrt(current_sales_var_interval) / current_sales_mean_interval) ** 2
                else:
                    current_cv2 = 1.0
            last_event_day = float(t)
        else:
            last_hawkes = current_hawkes_val

    feat_periodicity_score, feat_detected_period = compute_online_periodicity(
        sales,
        min_lag=min_period_lag,
        max_lag=max_period_lag,
        alpha=period_alpha,
        threshold_ratio=period_threshold_ratio,
        min_score=period_min_score
    )

    return feat_adi, feat_cv2, feat_hawkes, feat_days_since, feat_periodicity_score, feat_detected_period


def add_regime_features(
    df: pd.DataFrame,
    item_col: str = '書名',
    date_col: str = '日付',
    sales_col: str = 'POS販売冊数',
    hawkes_decay: float = 0.1,
    initial_adi: float = 30.0,
    initial_cv2: float = 1.0,
    base_alpha: float = 0.1,
    min_period_lag: int = 3,
    max_period_lag: int = 60,
    period_alpha: float = 0.05,
    period_threshold_ratio: float = 0.75,
    period_min_score: float = 0.25
) -> pd.DataFrame:
    """
    オンライン自己相関 + Red Noise対策付き周期性検出を含む特徴量を追加

    Args:
        min_period_lag: 最小周期（デフォルト3で2日AR(1)ノイズを除外）
        period_alpha: 周期性検出の学習率 (0.01-0.1推奨)
            - 小さい値 (0.01-0.03): 安定するが周期変化への反応が遅い
            - 大きい値 (0.07-0.1): 変化に敏感だがノイズの影響を受けやすい
        period_threshold_ratio: 倍音抑制の閾値比率 (0.65-0.85推奨)
            - 0.75 (推奨): バランス型、基本波と倍音を適切に識別
            - 0.85 (厳格): より強い倍音抑制
            - 0.65 (寛容): 弱い倍音抑制
        period_min_score: 最小相関スコア (0.15-0.35推奨)
            - 0.25 (推奨): バランス型、弱い相関を除外
            - 0.35 (厳格): より強力なノイズ除外、真の周期のみ
            - 0.15 (寛容): 緩やかな周期も許容
    """
    df = df.sort_values([item_col, date_col]).reset_index(drop=True)

    results = []

    for item_name, group in df.groupby(item_col, observed=True):
        sales_array = group[sales_col].values.astype(np.float32)

        feat_adi, feat_cv2, feat_hawkes, feat_days_since, feat_periodicity_score, feat_detected_period = compute_regime_features(
            sales_array,
            hawkes_decay=hawkes_decay,
            initial_adi=initial_adi,
            initial_cv2=initial_cv2,
            base_alpha=base_alpha,
            min_period_lag=min_period_lag,
            max_period_lag=max_period_lag,
            period_alpha=period_alpha,
            period_threshold_ratio=period_threshold_ratio,
            period_min_score=period_min_score
        )

        results.append(pd.DataFrame({
            'feat_adi': feat_adi,
            'feat_cv2': feat_cv2,
            'feat_hawkes': feat_hawkes,
            'feat_days_since': feat_days_since,
            'feat_periodicity_score': feat_periodicity_score,
            'feat_detected_period': feat_detected_period
        }, index=group.index))

    features_df = pd.concat(results, axis=0).sort_index()
    result_df = pd.concat([df, features_df], axis=1)

    return result_df


def add_regime_scores(
    df: pd.DataFrame,
    lambda_adi: float = 30.0,
    item_col: str = '書名'
) -> pd.DataFrame:
    """
    オンライン自己相関ベースの周期性スコアを含む最終スコアを計算
    """
    df = df.copy()

    df['score_sparse'] = 1.0 - np.exp(-df['feat_adi'] / lambda_adi)

    rolling_max_hawkes = df.groupby(item_col)['feat_hawkes'].transform(
        lambda x: x.expanding(min_periods=1).max()
    )
    scale_hawkes = rolling_max_hawkes.replace(0, 1.0) * 0.5
    df['score_burst'] = np.tanh(df['feat_hawkes'] / scale_hawkes)

    df['score_periodic'] = df['feat_periodicity_score'].clip(0.0, 1.0)

    return df


def plot_regime_patterns(
    df: pd.DataFrame,
    n_quantiles: int = 11,
    item_col: str = '書名',
    date_col: str = '日付',
    sales_col: str = 'POS販売冊数',
    output_dir: str = 'regime_plots_final'
):
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


@jit(nopython=True)
def calculate_z_score(sales: np.ndarray, context_length: int = 180) -> np.ndarray:
    n = len(sales)
    z_score_arr = np.zeros(n, dtype=np.float32)

    for i in range(n):
        if i > 0:
            start_local = max(0, i - context_length + 1)
            hist_sales_excl_today = sales[start_local : i]

            if len(hist_sales_excl_today) > 0:
                hist_mean = np.mean(hist_sales_excl_today)
                hist_std = np.std(hist_sales_excl_today)
                if hist_std > 0:
                    z_score_arr[i] = (sales[i] - hist_mean) / hist_std

    return z_score_arr


if __name__ == '__main__':
    df = pd.read_parquet('df_for.parquet')
    df['POS販売冊数'] = np.log1p(df['POS販売冊数'])

    print("データ読み込み完了...")
    print(f"商品数: {df['書名'].nunique()}, レコード数: {len(df)}")

    print("\n特徴量生成中（Red Noise対策付きオンライン自己相関）...")
    print("改良点:")
    print("  - min_period_lag=3: 2日AR(1)ノイズを除外")
    print("  - min_score=0.25: 弱い相関（偽周期）を除外")
    print("  - Strict Local Peak: 単調減衰を除外")

    df_with_features = add_regime_features(
        df,
        base_alpha=0.1,
        min_period_lag=3,
        max_period_lag=60,
        period_alpha=0.05,
        period_threshold_ratio=0.75,
        period_min_score=0.25
    )

    print("\nスコア計算中...")
    df_scored = add_regime_scores(df_with_features)

    print("\n=== 周期性検出の統計 ===")
    print(f"周期性スコア:")
    print(f"  平均: {df_scored['score_periodic'].mean():.3f}")
    print(f"  中央値: {df_scored['score_periodic'].median():.3f}")
    print(f"  最大: {df_scored['score_periodic'].max():.3f}")
    print(f"  高周期性（≥0.5）: {(df_scored['score_periodic'] >= 0.5).sum() / len(df_scored) * 100:.1f}%")

    detected_periods = df_scored[df_scored['feat_detected_period'] > 0]['feat_detected_period']
    if len(detected_periods) > 0:
        print(f"\n検出された周期:")
        print(f"  検出率: {len(detected_periods) / len(df_scored) * 100:.1f}%")
        print(f"  平均: {detected_periods.mean():.1f}日")
        print(f"  中央値: {detected_periods.median():.1f}日")
        print(f"  頻出周期: {detected_periods.mode().values[0]:.0f}日" if len(detected_periods.mode()) > 0 else "")

        period_counts = detected_periods.value_counts().head(10)
        print(f"\n  上位10周期:")
        for period, count in period_counts.items():
            print(f"    {period:.0f}日: {count:6d}回 ({count/len(detected_periods)*100:.1f}%)")

    print("\nプロット作成中...")
    plot_regime_patterns(df_scored, n_quantiles=11, output_dir='regime')
    print("完了")