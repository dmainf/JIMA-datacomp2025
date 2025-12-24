import pandas as pd
import numpy as np
from gluonts.dataset.pandas import PandasDataset
from gluonts.torch.model.tft import TemporalFusionTransformerEstimator
from gluonts.evaluation import make_evaluation_predictions, Evaluator
import warnings
import logging
import torch
import matplotlib.pyplot as plt
from pytorch_lightning.loggers import CSVLogger
import os
import shutil
from gluonts.torch.distributions import NegativeBinomialOutput, StudentTOutput

plt.rcParams['font.family'] = ['Hiragino Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

warnings.filterwarnings("ignore")
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
logging.getLogger("gluonts").setLevel(logging.ERROR)

PREDICTION_LENGTH = 45
CONTEXT_LENGTH = 180
EPOC = 200

ALL_TITLE_PREDICT = True
QUANTILES = [0.1, 0.25, 0.5, 0.75, 0.9]
USE_LOG_SCALE = False

static_cols = ['出版社', '著者名', '大分類', '中分類', '小分類']

def add_calendar_features(df):
    """
    カレンダー特徴量を追加（未来も既知）
    Sin/Cos変換を行い、時間の「周期性」を維持する
    """
    print("\n=== Adding Calendar Features (Cyclical & Known in Future) ===")

    month_norm = (df['日付'].dt.month - 1) / 12.0
    df['month_sin'] = np.sin(2 * np.pi * month_norm).astype(np.float32)
    df['month_cos'] = np.cos(2 * np.pi * month_norm).astype(np.float32)

    day_norm = (df['日付'].dt.day - 1) / 31.0
    df['day_sin'] = np.sin(2 * np.pi * day_norm).astype(np.float32)
    df['day_cos'] = np.cos(2 * np.pi * day_norm).astype(np.float32)

    dow_norm = df['日付'].dt.dayofweek / 7.0
    df['dow_sin'] = np.sin(2 * np.pi * dow_norm).astype(np.float32)
    df['dow_cos'] = np.cos(2 * np.pi * dow_norm).astype(np.float32)

    week_norm = (df['日付'].dt.isocalendar().week - 1) / 53.0
    df['week_sin'] = np.sin(2 * np.pi * week_norm).astype(np.float32)
    df['week_cos'] = np.cos(2 * np.pi * week_norm).astype(np.float32)

    df['is_weekend'] = (df['日付'].dt.dayofweek >= 5).astype(np.float32)
    df['is_month_start'] = (df['日付'].dt.day <= 5).astype(np.float32)
    df['is_month_end'] = (df['日付'].dt.day >= 25).astype(np.float32)

    print("Added: month_sin/cos, day_sin/cos, dow_sin/cos, week_sin/cos, flags")
    return df

def add_hazard_features(df, context_length=CONTEXT_LENGTH, prediction_length=PREDICTION_LENGTH):
    """
    【改良版】統計的スパイク検知機能を搭載
    Momentum、Volatility、Statistical Spike Detection、移動平均を追加
    """
    print("\n=== Adding Momentum, Volatility, Spike & Rolling Mean Features ===")
    df = df.sort_values(['書名', '日付']).reset_index(drop=True)
    df['target_shifted'] = df.groupby('書名')['POS販売冊数'].shift(1).fillna(0)

    rolling_short = df.groupby('書名')['target_shifted'].transform(lambda x: x.rolling(window=prediction_length, min_periods=1).mean())
    rolling_long = df.groupby('書名')['target_shifted'].transform(lambda x: x.rolling(window=context_length, min_periods=1).mean())

    df['momentum'] = rolling_short / (rolling_long + 1e-5)
    df['volatility'] = df.groupby('書名')['target_shifted'].transform(lambda x: x.rolling(window=context_length, min_periods=1).std()).fillna(0)

    roll_mean = df.groupby('書名')['POS販売冊数'].transform(lambda x: x.shift(1).rolling(window=context_length, min_periods=1).mean())
    roll_std = df.groupby('書名')['POS販売冊数'].transform(lambda x: x.shift(1).rolling(window=context_length, min_periods=1).std())

    z_score = (df['POS販売冊数'] - roll_mean) / (roll_std + 1e-5)
    df['deviation_score'] = 50 + (z_score * 10)

    SPIKE_DEVIATION_THRESHOLD = 75
    df['is_spike'] = (df['deviation_score'] >= SPIKE_DEVIATION_THRESHOLD) & (df['POS販売冊数'] > 0)
    df['last_spike_date'] = df.groupby('書名')['日付'].transform(
        lambda x: x.where(df.loc[x.index, 'is_spike']).ffill().shift(1)
    )
    df['days_since_last_spike'] = (df['日付'] - df['last_spike_date']).dt.days
    df['release_date'] = df.groupby('書名')['日付'].transform('min')
    df['days_since_release'] = (df['日付'] - df['release_date']).dt.days
    df['days_since_last_spike'] = df['days_since_last_spike'].fillna(df['days_since_release'])
    df['log_momentum'] = np.log1p(df['momentum']).astype(np.float32)
    df['log_volatility'] = np.log1p(df['volatility']).astype(np.float32)
    df['log_days_since_last_spike'] = np.log1p(df['days_since_last_spike']).astype(np.float32)

    df['log_rolling_mean_7d'] = np.log1p(
        df.groupby('書名')['target_shifted'].transform(lambda x: x.rolling(window=7, min_periods=1).mean())
    ).astype(np.float32)

    df['log_rolling_mean_30d'] = np.log1p(
        df.groupby('書名')['target_shifted'].transform(lambda x: x.rolling(window=30, min_periods=1).mean())
    ).astype(np.float32)

    drop_cols = [
        'release_date', 'target_shifted', 'momentum', 'volatility',
        'days_since_release', 'last_spike_date', 'days_since_last_spike',
        'is_spike', 'deviation_score'
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    print("Added: log_momentum, log_volatility, log_days_since_last_spike, log_rolling_mean_7d, log_rolling_mean_30d")
    return df


def extract_quartile_books(df):
    print("\n=== Identifying Representative Books (Quantiles) ===")
    total_sales = df.groupby('書名')['POS販売冊数'].sum().sort_values()
    quantiles = [0.0, 0.25, 0.5, 0.75, 1.0]
    quantile_names = ["Min (0%)", "Q1 (25%)", "Median (50%)", "Q3 (75%)", "Max (100%)"]
    quartile_books = {}
    print(f"{'Quantile':<15} | {'Total Sales':<12} | {'Book Name'}")
    print("-" * 60)
    num_books = len(total_sales)
    for q, name in zip(quantiles, quantile_names):
        idx = int((num_books - 1) * q)
        book_name = total_sales.index[idx]
        sales_val = total_sales.iloc[idx]
        quartile_books[book_name] = {
            "label": name,
            "total_sales": sales_val
        }
        print(f"{name:<15} | {sales_val:,.0f}冊       | {book_name}")
    return quartile_books


def scaling_data(df):
    if USE_LOG_SCALE:
        df['POS販売冊数'] = np.log1p(df['POS販売冊数']).astype(np.float32)
        print("\n=== Applied log1p transformation ===")
    else:
        df['POS販売冊数'] = df['POS販売冊数'].astype(np.float32)
        print("\n=== No scaling applied ===")

    print("\n=== POS Sales Statistics ===")
    print(f"Mean: {df['POS販売冊数'].mean():.4f}")
    print(f"Variance: {df['POS販売冊数'].var():.4f}")
    print(f"Std Dev: {df['POS販売冊数'].std():.4f}")


def prepare_dataset(df):
    df['id'] = df['書名'].astype(str)
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
        feat_dynamic_real=[
            'month_sin', 'month_cos',
            'day_sin', 'day_cos',
            'dow_sin', 'dow_cos',
            'week_sin', 'week_cos',
            'is_weekend',
            'is_month_start',
            'is_month_end'
        ],
        past_feat_dynamic_real=[
            'log_momentum',
            'log_volatility',
            'log_days_since_last_spike',
            'log_rolling_mean_7d',
            'log_rolling_mean_30d'
        ]
    )
    full_dataset = list(dataset)
    print(f"Number of time series: {len(full_dataset)}")
    print("\n=== Creating Training-Specific Dataset (No Data Leakage) ===")
    train_dataset = []
    for entry in full_dataset:
        train_entry = entry.copy()
        train_entry["target"] = entry["target"][:-PREDICTION_LENGTH]
        train_dataset.append(train_entry)
    print("Training dataset created by removing prediction length from the end of each series.")
    return full_dataset, train_dataset, cardinality


def plot_training_history():
    log_dir = 'lightning_logs/tft_training'
    save_path = 'figure/training_history.png'
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    metrics_file = None
    if not os.path.exists(log_dir):
        print(f"Log directory not found: {log_dir}")
        return

    version_dirs = [d for d in os.listdir(log_dir) if d.startswith('version_') and os.path.isdir(os.path.join(log_dir, d))]
    if not version_dirs:
        print("No lightning_logs version directory found.")
        return
    latest_version = sorted(version_dirs, key=lambda x: int(x.split('_')[1]))[-1]
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


def train_model(train_dataset, cardinality):
    csv_logger = CSVLogger("lightning_logs", name="tft_training")
    print("\n=== Training ===")

    if USE_LOG_SCALE:
        print("Distribution: StudentTOutput (Suitable for log-transformed continuous data)")
        distr_output = StudentTOutput()
    else:
        print("Distribution: NegativeBinomialOutput (Suitable for sparse count data)")
        distr_output = NegativeBinomialOutput()

    estimator = TemporalFusionTransformerEstimator(
        freq="D",
        prediction_length=PREDICTION_LENGTH,
        context_length=CONTEXT_LENGTH,
        static_cardinalities=cardinality,
        distr_output=distr_output,
        batch_size=64,
        trainer_kwargs={
            "max_epochs": EPOC,
            "accelerator": "auto",
            "logger": csv_logger,
            "enable_progress_bar": True,
            "enable_model_summary": False
        }
    )
    predictor = estimator.train(training_data=train_dataset)
    return predictor


def process_logs():
    log_dir = 'lightning_logs/tft_training'
    print("\n=== Plotting Training History ===")
    plot_training_history()
    print("\n=== Cleaning up logs ===")
    if os.path.exists(log_dir):
        version_dirs = [d for d in os.listdir(log_dir) if d.startswith('version_') and os.path.isdir(os.path.join(log_dir, d))]
        if version_dirs:
            latest_version = sorted(version_dirs, key=lambda x: int(x.split('_')[1]))[-1]
            latest_version_dir_path = os.path.join(log_dir, latest_version)
            for item in version_dirs:
                item_path = os.path.join(log_dir, item)
                if os.path.isdir(item_path) and item_path != latest_version_dir_path:
                    shutil.rmtree(item_path)
            print("Older lightning_logs versions removed")
        else:
            print("No versioned lightning_logs directories found to clean.")
    else:
        print("lightning_logs directory not found, no cleanup performed.")


def save_forecast(forecast, actual_data_log, save_path, title, use_log_scale, prediction_length, context_length):
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    start_timestamp = forecast.start_date.to_timestamp() if hasattr(forecast.start_date, 'to_timestamp') else pd.Timestamp(forecast.start_date)
    history_end_date = start_timestamp - pd.Timedelta(days=1)
    history_start_date = start_timestamp - pd.Timedelta(days=context_length)

    try:
        past_series = actual_data_log[history_start_date:history_end_date]
    except KeyError:
        actual_data_log.index = actual_data_log.index.to_timestamp() if hasattr(actual_data_log.index, 'to_timestamp') else actual_data_log.index
        past_series = actual_data_log[history_start_date:history_end_date]

    future_end_date = start_timestamp + pd.Timedelta(days=prediction_length - 1)
    target_series = actual_data_log[start_timestamp:future_end_date]

    if not past_series.empty:
        history_dates = past_series.index
        if hasattr(history_dates, 'to_timestamp'):
            history_dates = history_dates.to_timestamp()

        history_values = past_series.values
        if use_log_scale:
            history_values = np.expm1(history_values)
    else:
        history_dates = []
        history_values = []

    if not target_series.empty:
        target_dates = target_series.index
        if hasattr(target_dates, 'to_timestamp'):
            target_dates = target_dates.to_timestamp()

        target_values = target_series.values
        if use_log_scale:
            target_values = np.expm1(target_values)
    else:
        target_dates = []
        target_values = []

    forecast_dates = [start_timestamp + pd.Timedelta(days=i) for i in range(prediction_length)]

    plt.figure(figsize=(12, 6))

    if len(history_dates) > 0:
        plt.plot(history_dates, history_values, label="Actual History", color='black', linestyle='--')

    if len(target_dates) > 0:
        plt.plot(target_dates, target_values, label="Actual Target", color='black', marker='o', markersize=3)

    q_vals = {}
    for q in QUANTILES:
        val = forecast.quantile(q)
        q_vals[q] = np.expm1(val) if use_log_scale else val

    plt.fill_between(forecast_dates, q_vals[0.1], q_vals[0.9], color='blue', alpha=0.15, label="80% Interval (10-90%)")
    plt.fill_between(forecast_dates, q_vals[0.25], q_vals[0.75], color='cyan', alpha=0.5, label="IQR (25-75%)")
    plt.plot(forecast_dates, q_vals[0.5], label="Median (50%)", color='blue', linewidth=2)

    plt.title(title, fontsize=14)
    plt.xlabel("日付")
    plt.ylabel("POS販売冊数 (expm1)" if use_log_scale else "POS販売冊数")
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


def process_predictions(forecasts, tss, quartile_books, df):
    print("\n=== Calculating Decile Groups ===")
    total_sales = df.groupby('書名')['POS販売冊数'].sum().sort_values(ascending=False)
    decile_groups = {}
    n_books = len(total_sales)
    for i, book_name in enumerate(total_sales.index):
        decile = int(i / n_books * 10) + 1
        if decile > 10:
            decile = 10
        decile_groups[book_name] = decile

    total_sales_all = total_sales.sum()

    for decile in range(1, 11):
        if decile == 1:
            folder = "top_10"
        elif decile == 10:
            folder = "90-100"
        else:
            folder = f"{(decile-1)*10}-{decile*10}"
        os.makedirs(f'figure/predict/{folder}', exist_ok=True)
        count = sum(1 for v in decile_groups.values() if v == decile)

        decile_sales = sum(total_sales[book] for book, d in decile_groups.items() if d == decile)
        decile_pct = (decile_sales / total_sales_all) * 100

        print(f"  {folder}: {count} books, {decile_pct:.1f}% of total sales")

    print("\n=== Saving Predictions ===")
    if ALL_TITLE_PREDICT:
        print("Mode: Saving ALL book predictions")
    else:
        print("Mode: Saving ONLY representative book predictions (min, quartiles, max)")

    saved_count = 0
    for i, forecast in enumerate(forecasts):
        book_id = forecast.item_id
        actual_data_log = tss[i]
        safe_book_id = book_id.replace("/", "_").replace("\\", "_")

        decile = decile_groups.get(book_id, 10)
        if decile == 1:
            folder = "top_10"
        elif decile == 10:
            folder = "90-100"
        else:
            folder = f"{(decile-1)*10}-{decile*10}"

        if ALL_TITLE_PREDICT:
            base_save_path = f"figure/predict/{folder}/forecast_{safe_book_id}.png"
            save_forecast(
                forecast=forecast,
                actual_data_log=actual_data_log,
                save_path=base_save_path,
                title=f"{book_id}",
                use_log_scale=USE_LOG_SCALE,
                prediction_length=PREDICTION_LENGTH,
                context_length=CONTEXT_LENGTH
            )
            saved_count += 1

        if book_id in quartile_books:
            info = quartile_books[book_id]
            label = info["label"]
            total_sales = info["total_sales"]
            print(f"Plotting Representative: {label} - {book_id}")
            safe_label = label.replace(" ", "").replace("(", "").replace(")", "").replace("%", "")
            rep_save_path = f"figure/forecast_{safe_label}_{book_id}.png"
            save_forecast(
                forecast=forecast,
                actual_data_log=actual_data_log,
                save_path=rep_save_path,
                title=f"[{label}] {book_id}\nTotal Sales: {total_sales:,.0f} books",
                use_log_scale=USE_LOG_SCALE,
                prediction_length=PREDICTION_LENGTH,
                context_length=CONTEXT_LENGTH
            )
            if not ALL_TITLE_PREDICT:
                saved_count += 1
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(forecasts)}")

    if ALL_TITLE_PREDICT:
        print(f"All {len(forecasts)} prediction plots saved to figure/predict/*/")
    else:
        print(f"{saved_count} representative prediction plots saved to figure/")


def main():
    print("=== Loading Data ===")
    df = pd.read_parquet('data/df_for.parquet')
    print("Complete!")

    df = add_calendar_features(df)
    df = add_hazard_features(df, CONTEXT_LENGTH)
    quartile_books = extract_quartile_books(df)
    scaling_data(df)
    full_dataset, train_dataset, cardinality = prepare_dataset(df)
    predictor = train_model(train_dataset, cardinality)
    process_logs()

    forecasts, tss = evaluate_prediction(predictor, full_dataset)
    process_predictions(forecasts, tss, quartile_books, df)

    print("\nAll plots saved. Process Complete.")

if __name__ == "__main__":
    main()
