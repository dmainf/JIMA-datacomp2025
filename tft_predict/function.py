import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from gluonts.dataset.pandas import PandasDataset
from gluonts.torch.model.tft import TemporalFusionTransformerEstimator
from gluonts.evaluation import make_evaluation_predictions, Evaluator
from gluonts.torch.distributions import QuantileOutput
from pytorch_lightning.loggers import CSVLogger
import os
import shutil
from numba import jit
from typing import Tuple

plt.rcParams['font.family'] = ['Hiragino Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


def _get_latest_version(log_dir):
    version_dirs = [d for d in os.listdir(log_dir)
                    if d.startswith('version_') and os.path.isdir(os.path.join(log_dir, d))]
    return sorted(version_dirs, key=lambda x: int(x.split('_')[1]))[-1] if version_dirs else None


def _get_decile_folder(decile):
    if decile == 1:
        return "top_10"
    elif decile == 10:
        return "90-100"
    return f"{(decile-1)*10}-{decile*10}"


def extract_decile_books(df):
    print("\n=== Identifying Representative Books (Deciles - 10% intervals) ===")
    total_sales = df.groupby('書名')['POS販売冊数'].sum().sort_values()
    quantiles = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    quantile_names = ["Min (0%)", "D1 (10%)", "D2 (20%)", "D3 (30%)", "D4 (40%)",
                      "Median (50%)", "D6 (60%)", "D7 (70%)", "D8 (80%)", "D9 (90%)", "Max (100%)"]
    decile_books = {}
    print(f"{'Decile':<15} | {'Total Sales':<12} | {'Book Name'}")
    print("-" * 60)
    num_books = len(total_sales)
    for q, name in zip(quantiles, quantile_names):
        idx = int((num_books - 1) * q)
        book_name = total_sales.index[idx]
        sales_val = total_sales.iloc[idx]
        decile_books[book_name] = {
            "label": name,
            "total_sales": sales_val
        }
        print(f"{name:<15} | {sales_val:,.0f}冊       | {book_name}")
    return decile_books


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


def calculate_hazard_features(
    df,
    context_length=180,
    recent_period=14,
    spike_thresholds=[2.0, 2.5, 3.0],
    window_size=5,
    hawkes_decay=0.1,
    initial_adi=30.0,
    initial_cv2=1.0,
    lambda_adi=30.0,
    scale_hawkes=10.0
):
    print("\n=== Calculating Hazard Features & Regime Classification ===")
    print(f"Context length: {context_length} days")
    print(f"Recent period (momentum): {recent_period} days")
    print(f"Spike thresholds: {spike_thresholds}")
    print(f"Regime parameters: window_size={window_size}, hawkes_decay={hawkes_decay}")
    print("Using vectorized operations for better performance")

    df = df.sort_values(['書名', '日付']).reset_index(drop=True)

    rolling_mean_context = df.groupby('書名')['POS販売冊数'].transform(
        lambda x: x.shift(1).rolling(window=context_length, min_periods=1).mean()
    )
    rolling_mean_recent = df.groupby('書名')['POS販売冊数'].transform(
        lambda x: x.shift(1).rolling(window=recent_period, min_periods=1).mean()
    )
    rolling_std = df.groupby('書名')['POS販売冊数'].transform(
        lambda x: x.shift(1).rolling(window=context_length, min_periods=1).std()
    )

    df['momentum'] = (rolling_mean_recent / np.maximum(rolling_mean_context, 1e-8)).fillna(1.0).astype(np.float32)
    df['volatility'] = rolling_std.fillna(0.0).astype(np.float32)
    df['z_score'] = (
        (df['POS販売冊数'] - rolling_mean_context) / np.maximum(rolling_std, 1e-8)
    ).fillna(0.0).replace([np.inf, -np.inf], 0.0).clip(-10, 10).astype(np.float32)

    def calculate_days_since_spike(group, threshold):
        is_spike = group['z_score'] >= threshold
        indices = pd.Series(np.arange(len(group)), index=group.index)
        spike_indices = indices.where(is_spike)
        last_spike_idx = spike_indices.ffill()
        days_since = indices - last_spike_idx
        return days_since.fillna(0).astype(np.int32)

    for threshold in spike_thresholds:
        col_name = f'days_since_spike_{threshold}'
        df[col_name] = df.groupby('書名', group_keys=False).apply(
            lambda g: calculate_days_since_spike(g, threshold)
        ).astype(np.int32)

    print("\n=== Calculating Regime Features ===")
    regime_results = []
    for item_name, group in df.groupby('書名', observed=True):
        sales_array = group['POS販売冊数'].values.astype(np.float32)

        feat_adi, feat_cv2, feat_hawkes, feat_days_since = compute_regime_features(
            sales_array,
            window_size=window_size,
            hawkes_decay=hawkes_decay,
            initial_adi=initial_adi,
            initial_cv2=initial_cv2
        )

        regime_results.append(pd.DataFrame({
            'feat_adi': feat_adi,
            'feat_cv2': feat_cv2,
            'feat_hawkes': feat_hawkes,
            'feat_days_since': feat_days_since
        }, index=group.index))

    regime_features_df = pd.concat(regime_results, axis=0).sort_index()
    df = pd.concat([df, regime_features_df], axis=1)

    print("\n=== Calculating Regime Scores ===")
    df['score_sparse'] = (1.0 - np.exp(-df['feat_adi'] / lambda_adi)).astype(np.float32)
    df['score_periodic'] = np.clip(1.0 - np.sqrt(df['feat_cv2']), 0.0, 1.0).astype(np.float32)
    df['score_burst'] = np.tanh(df['feat_hawkes'] / scale_hawkes).astype(np.float32)

    print("\n=== Hazard Features Statistics ===")
    print(f"Momentum - Mean: {df['momentum'].mean():.4f}, Std: {df['momentum'].std():.4f}")
    print(f"Volatility - Mean: {df['volatility'].mean():.4f}, Std: {df['volatility'].std():.4f}")
    print(f"Z-score - Mean: {df['z_score'].mean():.4f}, Std: {df['z_score'].std():.4f}")
    for threshold in spike_thresholds:
        col_name = f'days_since_spike_{threshold}'
        print(f"Days since spike ({threshold}) - Mean: {df[col_name].mean():.1f}, Median: {df[col_name].median():.1f}")

    print("\n=== Regime Features Statistics ===")
    print(f"ADI - Mean: {df['feat_adi'].mean():.4f}, Std: {df['feat_adi'].std():.4f}")
    print(f"CV2 - Mean: {df['feat_cv2'].mean():.4f}, Std: {df['feat_cv2'].std():.4f}")
    print(f"Hawkes - Mean: {df['feat_hawkes'].mean():.4f}, Std: {df['feat_hawkes'].std():.4f}")
    print(f"Days Since Event - Mean: {df['feat_days_since'].mean():.4f}, Median: {df['feat_days_since'].median():.4f}")

    print("\n=== Regime Scores Statistics ===")
    print(f"Sparse Score - Mean: {df['score_sparse'].mean():.4f}, Std: {df['score_sparse'].std():.4f}")
    print(f"Periodic Score - Mean: {df['score_periodic'].mean():.4f}, Std: {df['score_periodic'].std():.4f}")
    print(f"Burst Score - Mean: {df['score_burst'].mean():.4f}, Std: {df['score_burst'].std():.4f}")

    return df


def scaling_data(df, use_log_scale):
    if use_log_scale:
        df['POS販売冊数'] = np.log1p(df['POS販売冊数']).astype(np.float32)
        print("\n=== Applied log1p transformation to POS販売冊数 ===")

        df['log_本体価格'] = np.log1p(df['本体価格']).astype(np.float32)
        df = df.drop('本体価格', axis=1)
        print(f"\n=== Applied log1p transformation to 本体価格 ===")
        print(f"Created 'log_本体価格' column and removed original '本体価格' column")
        print(f"Log price range: {df['log_本体価格'].min():.4f} - {df['log_本体価格'].max():.4f}")
    else:
        df['log_本体価格'] = df['本体価格'].astype(np.float32)
        df = df.drop('本体価格', axis=1)
        print("\n=== No scaling applied ===")
        print("Renamed '本体価格' to 'log_本体価格'")

    print("\n=== POS Sales Statistics ===")
    print(f"Mean: {df['POS販売冊数'].mean():.4f}")
    print(f"Variance: {df['POS販売冊数'].var():.4f}")
    print(f"Std Dev: {df['POS販売冊数'].std():.4f}")

    return df


def calculate_temporal_relative_features(df, target_col, category_cols=['大分類', '中分類', '小分類'], context_length=180):
    print(f"\n=== Calculating Temporal Relative Features for {target_col} ===")
    print(f"Category columns: {category_cols}")
    print(f"Context length: {context_length} days (using past data only to prevent data leakage)")

    df_result = df.copy()

    if target_col not in df_result.columns:
        print(f"Warning: {target_col} not found in dataframe, skipping...")
        return df_result

    print(f"{target_col} range: min={df_result[target_col].min()}, max={df_result[target_col].max()}")

    df_result = df_result.sort_values(['書名', '日付']).reset_index(drop=True)

    for category_col in category_cols:
        if category_col not in df_result.columns:
            print(f"Warning: {category_col} not found in dataframe, skipping...")
            continue

        print(f"\n--- Processing {category_col} for {target_col} ---")

        relative_col = f'{category_col}_{target_col}_relative'
        z_score_col = f'{category_col}_{target_col}_z_score'
        category_daily = df_result.groupby([category_col, '日付'], observed=False)[target_col].mean().reset_index()
        category_daily = category_daily.rename(columns={target_col: f'{target_col}_category_mean'})
        category_daily = category_daily.sort_values([category_col, '日付'])
        category_daily['category_rolling_mean'] = category_daily.groupby(category_col, observed=False)[f'{target_col}_category_mean'].transform(
            lambda x: x.shift(1).rolling(window=context_length, min_periods=1).mean()
        )

        category_daily['category_rolling_std'] = category_daily.groupby(category_col, observed=False)[f'{target_col}_category_mean'].transform(
            lambda x: x.shift(1).rolling(window=context_length, min_periods=1).std()
        )
        df_result = df_result.merge(
            category_daily[[category_col, '日付', 'category_rolling_mean', 'category_rolling_std']],
            on=[category_col, '日付'],
            how='left'
        )
        df_result[relative_col] = (
            df_result[target_col] - df_result['category_rolling_mean']
        ).fillna(0).astype(np.float32)

        df_result[z_score_col] = (
            (df_result[target_col] - df_result['category_rolling_mean']) /
            np.maximum(df_result['category_rolling_std'], 1e-8)
        ).replace([np.inf, -np.inf], np.nan).fillna(0).clip(-10, 10).astype(np.float32)
        df_result = df_result.drop(['category_rolling_mean', 'category_rolling_std'], axis=1)
        print(f"Created features:")
        print(f"  - {relative_col} (Mean: {df_result[relative_col].mean():.4f}, Std: {df_result[relative_col].std():.4f})")
        print(f"  - {z_score_col} (Mean: {df_result[z_score_col].mean():.4f}, Std: {df_result[z_score_col].std():.4f})")

    print(f"\n=== Temporal Relative Features for {target_col} Complete ===")
    return df_result


def calculate_static_relative_features(df, target_col, category_cols=['大分類', '中分類', '小分類']):
    print(f"\n=== Calculating Static Relative Features for {target_col} ===")
    print(f"Category columns: {category_cols}")

    unique_books = df[['書名', target_col] + category_cols].drop_duplicates()

    for cat_col in category_cols:
        if cat_col not in unique_books.columns:
            print(f"Warning: {cat_col} not found in dataframe, skipping...")
            continue

        print(f"\n--- Processing {cat_col} for {target_col} ---")

        grp = unique_books.groupby(cat_col, observed=False)[target_col]
        mean_val = grp.transform('mean')
        std_val = grp.transform('std').fillna(1.0)

        relative_col = f'{cat_col}_{target_col}_relative'
        z_score_col = f'{cat_col}_{target_col}_z_score'

        unique_books[relative_col] = (unique_books[target_col] - mean_val).astype(np.float32)
        unique_books[z_score_col] = ((unique_books[target_col] - mean_val) / np.maximum(std_val, 1e-8)).astype(np.float32)

        print(f"Created features:")
        print(f"  - {relative_col} (Mean: {unique_books[relative_col].mean():.4f}, Std: {unique_books[relative_col].std():.4f})")
        print(f"  - {z_score_col} (Mean: {unique_books[z_score_col].mean():.4f}, Std: {unique_books[z_score_col].std():.4f})")

    static_features = [c for c in unique_books.columns if f'_{target_col}_' in c]
    df_result = df.merge(unique_books[['書名'] + static_features], on='書名', how='left')

    print(f"\nCreated static features: {static_features}")
    print(f"=== Static Relative Features for {target_col} Complete ===")
    return df_result


def verify_features(df, feature_list, feature_category):
    print(f"\n=== Verifying {feature_category} Features ===")
    missing_features = []
    existing_features = []
    for feature in feature_list:
        if feature in df.columns:
            existing_features.append(feature)
        else:
            missing_features.append(feature)
    if existing_features:
        print(f"Found {len(existing_features)}/{len(feature_list)} features:")
        for feature in existing_features:
            print(f"  ✓ {feature}")
    if missing_features:
        print(f"\nMISSING {len(missing_features)} features:")
        for feature in missing_features:
            print(f"  ✗ {feature}")
        raise ValueError(f"Missing required features: {missing_features}")
    print(f"All {len(feature_list)} {feature_category} features verified successfully!")
    return True


def verify_all_features(df, static_cols, time_cols, relative_cols, hazard_cols):
    verify_features(df, static_cols, "Static")
    verify_features(df, time_cols, "Time")
    verify_features(df, relative_cols, "Relative")
    verify_features(df, hazard_cols, "Hazard")
    past_dynamic_cols = relative_cols + hazard_cols
    print(f"\n=== Feature Summary ===")
    print(f"Static features: {len(static_cols)}")
    print(f"Time features (known covariates): {len(time_cols)}")
    print(f"Relative features: {len(relative_cols)}")
    print(f"Hazard features: {len(hazard_cols)}")
    print(f"Total past dynamic features: {len(past_dynamic_cols)}")
    print(f"Total features: {len(static_cols) + len(time_cols) + len(past_dynamic_cols)}")


def save_training_data(df, filepath='train.parquet'):
    print("\n=== Saving Training Data ===")
    df.to_parquet(filepath, index=False)
    print(f"Training data saved to: {filepath}")
    print(f"Shape: {df.shape}")
    print(f"Columns: {len(df.columns)}")


def prepare_dataset(df, static_cols, time_feature_cols, past_dynamic_cols, prediction_length):
    print(f"\n=== Preparing Dataset ===")
    print(f"Static features: {static_cols}")
    print(f"Time features (known covariates): {time_feature_cols}")
    print(f"Past dynamic features (observed covariates): {past_dynamic_cols}")

    df['id'] = df['書名'].astype(str)
    df = df.sort_values(['id', '日付']).reset_index(drop=True)

    for col in static_cols:
        if df[col].dtype.name == 'category':
            df[col] = df[col].cat.remove_unused_categories()

    static_df = df.groupby('id', observed=False)[static_cols].first().reset_index()
    cardinality = [static_df[col].cat.categories.size for col in static_cols]
    print(f"Cardinality: {dict(zip(static_cols, cardinality))}")
    static_df_formatted = static_df[['id'] + static_cols].set_index('id')

    dataset = PandasDataset.from_long_dataframe(
        df,
        target='POS販売冊数',
        item_id='id',
        timestamp='日付',
        freq='D',
        static_features=static_df_formatted,
        feat_dynamic_real=time_feature_cols,
        past_feat_dynamic_real=past_dynamic_cols
    )
    full_dataset = list(dataset)
    print(f"Number of time series: {len(full_dataset)}")
    print("\n=== Creating Training-Specific Dataset (No Data Leakage) ===")
    train_dataset = []
    for entry in full_dataset:
        train_entry = entry.copy()
        train_entry["target"] = entry["target"][:-prediction_length]
        if "feat_dynamic_real" in entry:
            train_entry["feat_dynamic_real"] = entry["feat_dynamic_real"][:, :-prediction_length]
        if "past_feat_dynamic_real" in entry:
            train_entry["past_feat_dynamic_real"] = entry["past_feat_dynamic_real"][:, :-prediction_length]
        train_dataset.append(train_entry)
    print("Training dataset created by removing prediction length from the end of each series.")
    return full_dataset, train_dataset, cardinality


def plot_training_history():
    log_dir = 'lightning_logs/tft_training'
    save_path = 'valid/training_history.png'
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    if not os.path.exists(log_dir):
        print(f"Log directory not found: {log_dir}")
        return
    latest_version = _get_latest_version(log_dir)
    if not latest_version:
        print("No lightning_logs version directory found.")
        return
    metrics_file = os.path.join(log_dir, latest_version, 'metrics.csv')
    if not os.path.exists(metrics_file):
        print("No metrics file found in the latest version.")
        return
    print(f"Reading metrics from: {metrics_file}")
    df = pd.read_csv(metrics_file)
    plt.figure(figsize=(12, 6))
    if 'train_loss' in df.columns:
        train_df = df[['epoch', 'train_loss']].dropna()
        if not train_df.empty:
            plt.plot(train_df['epoch'], train_df['train_loss'], label='Train Loss', marker='o', linewidth=2)
    if 'val_loss' in df.columns:
        val_df = df[['epoch', 'val_loss']].dropna()
        if not val_df.empty:
            plt.plot(val_df['epoch'], val_df['val_loss'], label='Validation Loss', marker='s', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training history', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training history plot saved to: {save_path}")


def train_model(train_dataset, cardinality, prediction_length, context_length, epochs, quantiles):
    csv_logger = CSVLogger("lightning_logs", name="tft_training")
    print("\n=== Training ===")
    print("Distribution: QuantileOutput (Non-parametric, flexible for long-tail and intermittent demand)")
    print(f"Quantiles: {quantiles}")

    batch_size = 128
    num_workers = os.cpu_count() or 4
    print(f"Training config: batch_size={batch_size}, num_workers={num_workers}")

    distr_output = QuantileOutput(quantiles=quantiles)
    estimator = TemporalFusionTransformerEstimator(
        freq="D",
        prediction_length=prediction_length,
        context_length=context_length,
        static_cardinalities=cardinality,
        distr_output=distr_output,
        batch_size=batch_size,
        trainer_kwargs={
            "max_epochs": epochs,
            "accelerator": "auto",
            "logger": csv_logger,
            "enable_progress_bar": True,
            "enable_model_summary": False
        }
    )
    predictor = estimator.train(training_data=train_dataset, num_workers=num_workers)
    return predictor


def process_logs():
    log_dir = 'lightning_logs/tft_training'
    print("\n=== Plotting Training History ===")
    plot_training_history()
    print("\n=== Cleaning up logs ===")
    if not os.path.exists(log_dir):
        print("lightning_logs directory not found, no cleanup performed.")
        return
    latest_version = _get_latest_version(log_dir)
    if not latest_version:
        print("No versioned lightning_logs directories found to clean.")
        return
    latest_version_dir_path = os.path.join(log_dir, latest_version)
    version_dirs = [d for d in os.listdir(log_dir)
                    if d.startswith('version_') and os.path.isdir(os.path.join(log_dir, d))]
    for item in version_dirs:
        item_path = os.path.join(log_dir, item)
        if item_path != latest_version_dir_path:
            shutil.rmtree(item_path)
    print("Older lightning_logs versions removed")


def save_forecast(forecast, actual_data_log, save_path, title, use_log_scale, prediction_length, context_length, quantiles):
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    start_timestamp = forecast.start_date.to_timestamp() if hasattr(forecast.start_date, 'to_timestamp') else pd.Timestamp(forecast.start_date)
    history_end_date = start_timestamp - pd.Timedelta(days=1)
    history_start_date = start_timestamp - pd.Timedelta(days=context_length)
    if hasattr(actual_data_log.index, 'to_timestamp'):
        actual_data_log.index = actual_data_log.index.to_timestamp()
    past_series = actual_data_log[history_start_date:history_end_date]
    future_end_date = start_timestamp + pd.Timedelta(days=prediction_length - 1)
    target_series = actual_data_log[start_timestamp:future_end_date]
    history_dates = past_series.index if not past_series.empty else []
    history_values = past_series.values if not past_series.empty else []
    target_dates = target_series.index if not target_series.empty else []
    target_values = target_series.values if not target_series.empty else []
    forecast_dates = [start_timestamp + pd.Timedelta(days=i) for i in range(prediction_length)]
    plt.figure(figsize=(12, 6))
    if len(history_dates) > 0:
        plt.plot(history_dates, history_values, label="Actual History", color='black', linestyle='--')
    if len(target_dates) > 0:
        plt.plot(target_dates, target_values, label="Actual Target", color='black', marker='o', markersize=3)
    q_vals = {q: forecast.quantile(q) for q in quantiles}

    quantile_set = set(quantiles)
    pairs = []
    for q in sorted(quantiles):
        if q < 0.5:
            complement = round(1 - q, 2)
            if complement in quantile_set:
                pairs.append((q, complement))

    colors = ['blue', 'cyan', 'lightgreen', 'yellow']
    alphas = [0.15, 0.3, 0.45, 0.6]
    for idx, (lower, upper) in enumerate(pairs):
        coverage = int((upper - lower) * 100)
        color = colors[idx % len(colors)]
        alpha = alphas[idx % len(alphas)]
        plt.fill_between(forecast_dates, q_vals[lower], q_vals[upper],
                        color=color, alpha=alpha,
                        label=f"{coverage}% Interval ({int(lower*100)}-{int(upper*100)}%)")

    if 0.5 in q_vals:
        plt.plot(forecast_dates, q_vals[0.5], label="Median (50%)", color='blue', linewidth=2)
    plt.title(title, fontsize=14)
    plt.xlabel("日付")
    plt.ylabel("POS販売冊数 (log1p)" if use_log_scale else "POS販売冊数")
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.5)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def evaluate_prediction(predictor, full_dataset):
    print("\n=== Evaluating ===")
    forecast_it, ts_it = make_evaluation_predictions(
        dataset=full_dataset,
        predictor=predictor,
        num_samples=100
    )
    forecasts = list(forecast_it)
    tss = list(ts_it)
    evaluator = Evaluator()
    agg_metrics, item_metrics = evaluator(tss, forecasts)
    print("\n=== Metrics ===")
    for key, value in agg_metrics.items():
        print(f"{key}: {value:.4f}")
    return forecasts, tss


def process_predictions(forecasts, tss, decile_books, df, all_title_predict, use_log_scale, prediction_length, context_length, quantiles):
    print("\n=== Calculating Decile Groups ===")
    total_sales = df.groupby('書名')['POS販売冊数'].sum().sort_values(ascending=False)
    n_books = len(total_sales)
    decile_groups = {book_name: min(int(i / n_books * 10) + 1, 10)
                     for i, book_name in enumerate(total_sales.index)}
    total_sales_all = total_sales.sum()
    decile_sales_dict = {d: sum(total_sales[book] for book, dd in decile_groups.items() if dd == d)
                         for d in range(1, 11)}
    for decile in range(1, 11):
        folder = _get_decile_folder(decile)
        os.makedirs(f'valid/All_predict/{folder}', exist_ok=True)
        count = sum(1 for v in decile_groups.values() if v == decile)
        decile_pct = (decile_sales_dict[decile] / total_sales_all) * 100
        print(f"  {folder}: {count} books, {decile_pct:.1f}% of total sales")
    print("\n=== Saving Predictions ===")
    print("Mode: Saving ALL book predictions" if all_title_predict else "Mode: Saving ONLY representative book predictions (min, deciles, max)")

    saved_count = 0
    for i, forecast in enumerate(forecasts):
        book_id = forecast.item_id
        actual_data_log = tss[i]
        safe_book_id = book_id.replace("/", "_").replace("\\", "_")
        decile = decile_groups.get(book_id, 10)
        folder = _get_decile_folder(decile)
        if all_title_predict:
            base_save_path = f"valid/All_predict/{folder}/forecast_{safe_book_id}.png"
            save_forecast(
                forecast=forecast,
                actual_data_log=actual_data_log,
                save_path=base_save_path,
                title=f"{book_id}",
                use_log_scale=use_log_scale,
                prediction_length=prediction_length,
                context_length=context_length,
                quantiles=quantiles
            )
            saved_count += 1
        if book_id in decile_books:
            info = decile_books[book_id]
            label = info["label"]
            total_sales = info["total_sales"]
            print(f"Plotting Representative: {label} - {book_id}")
            safe_label = label.replace(" ", "").replace("(", "").replace(")", "").replace("%", "")
            rep_save_path = f"valid/forecast_{safe_label}_{book_id}.png"
            save_forecast(
                forecast=forecast,
                actual_data_log=actual_data_log,
                save_path=rep_save_path,
                title=f"[{label}] {book_id}\nTotal Sales: {total_sales:,.0f} books",
                use_log_scale=use_log_scale,
                prediction_length=prediction_length,
                context_length=context_length,
                quantiles=quantiles
            )
            if not all_title_predict:
                saved_count += 1
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(forecasts)}")
    print(f"All {len(forecasts)} prediction plots saved to valid/All_predict/*/" if all_title_predict
          else f"{saved_count} representative prediction plots saved to valid/")
