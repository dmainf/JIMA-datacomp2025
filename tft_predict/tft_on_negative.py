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
from gluonts.torch.distributions import NegativeBinomialOutput
from gluonts.time_feature import time_features_from_frequency_str

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
USE_LOG_SCALE = False
QUANTILES = [0.1, 0.25, 0.5, 0.75, 0.9]

static_cols = ['出版社', '著者名', '大分類', '中分類', '小分類']

class InterpretabilityCollector:
    def __init__(self, model, save_dir='figure/interpretability'):
        self.model = model
        self.save_dir = save_dir
        self.static_weights_list = []
        self.past_weights_list = []
        self.future_weights_list = []
        self.temporal_weights_list = []
        self.hooks = []
        os.makedirs(save_dir, exist_ok=True)

    def _hook_static(self, module, input, output):
        if isinstance(output, tuple) and len(output) > 1:
            w = output[1].detach().cpu().numpy()
            for i in range(w.shape[0]):
                self.static_weights_list.append(w[i].flatten())

    def _hook_past(self, module, input, output):
        if isinstance(output, tuple) and len(output) > 1:
            w = output[1].detach().cpu().numpy()
            w_mean_time = w.mean(axis=1)
            for i in range(w.shape[0]):
                self.past_weights_list.append(w_mean_time[i].flatten())

    def _hook_future(self, module, input, output):
        if isinstance(output, tuple) and len(output) > 1:
            w = output[1].detach().cpu().numpy()
            w_mean_time = w.mean(axis=1)
            for i in range(w.shape[0]):
                self.future_weights_list.append(w_mean_time[i].flatten())

    def _hook_temporal(self, module, input, output):
        if isinstance(output, tuple) and len(output) > 1:
            w = output[1].detach().cpu().numpy()
            for i in range(w.shape[0]):
                self.temporal_weights_list.append(w[i])

    def register(self):
        print("Registering hooks for interpretability...")
        self.hooks.append(self.model.static_selector.register_forward_hook(self._hook_static))
        self.hooks.append(self.model.ctx_selector.register_forward_hook(self._hook_past))
        self.hooks.append(self.model.tgt_selector.register_forward_hook(self._hook_future))
        self.hooks.append(self.model.temporal_decoder.attention.register_forward_hook(self._hook_temporal))

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []
        print("Hooks removed.")

    def save_and_plot_summary(self, static_col_names, time_feat_names, context_length, item_ids=None):
        if not self.static_weights_list:
            print("No interpretability data collected.")
            return

        print(f"\n=== Saving Interpretability Data to {self.save_dir} ===")

        all_static = np.array(self.static_weights_list)
        mean_static = all_static.mean(axis=0)

        plot_static = mean_static
        plot_names = static_col_names

        if len(mean_static) == len(static_col_names) + 1:
            plot_static = mean_static[1:]
        elif len(mean_static) != len(static_col_names):
            print(f"Warning: Static feature count mismatch. Expected {len(static_col_names)}, got {len(mean_static)}")
            plot_names = [f"Feat {i}" for i in range(len(mean_static))]

        self._plot_bar(
            plot_static, plot_names,
            "Average Static Variable Importance", "mean_static_importance.png"
        )
        np.save(os.path.join(self.save_dir, 'all_static_importance.npy'), all_static)

        if self.past_weights_list:
            all_past = np.array(self.past_weights_list)
            mean_past = all_past.mean(axis=0)

            past_names = ["Target(Log)"] + time_feat_names + ["Static_Context"]

            if len(mean_past) != len(past_names):
                 past_names = [f"Var {i}" for i in range(len(mean_past))]

            self._plot_bar(
                mean_past, past_names,
                "Average Past Variable Importance", "mean_past_importance.png"
            )
            np.save(os.path.join(self.save_dir, 'all_past_importance.npy'), all_past)

        if self.future_weights_list:
            all_future = np.array(self.future_weights_list)
            mean_future = all_future.mean(axis=0)

            future_names = time_feat_names + ["Static_Context"]

            if len(mean_future) != len(future_names):
                 future_names = [f"Var {i}" for i in range(len(mean_future))]

            self._plot_bar(
                mean_future, future_names,
                "Average Future Variable Importance", "mean_future_importance.png"
            )
            np.save(os.path.join(self.save_dir, 'all_future_importance.npy'), all_future)

        if self.temporal_weights_list:
            self._save_individual_attention_maps(self.temporal_weights_list, context_length, item_ids)

        print("Saved interpretability plots and data.")

    def _plot_bar(self, values, names, title, filename):
        plt.figure(figsize=(10, 6))
        indices = np.argsort(values)[::-1]

        try:
            sorted_values = values[indices]
            sorted_names = np.array(names)[indices]
        except IndexError as e:
            print(f"Error plotting {title}: {e}")
            print(f"Values shape: {values.shape}, Names len: {len(names)}")
            plt.close()
            return

        plt.bar(sorted_names, sorted_values, color='skyblue', edgecolor='black')
        plt.title(title, fontsize=14)
        plt.ylabel("Importance Score")
        plt.grid(axis='y', alpha=0.5)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, filename))
        plt.close()

    def _save_individual_attention_maps(self, temporal_weights_list, context_length, item_ids=None):
        individual_dir = os.path.join(self.save_dir, 'individual_attention_maps')
        os.makedirs(individual_dir, exist_ok=True)

        print(f"Saving {len(temporal_weights_list)} individual attention maps as PNG...")

        sum_attention = None
        count = 0

        for i, attn_map in enumerate(temporal_weights_list):
            if attn_map.ndim == 3:
                attn_map = attn_map.mean(axis=0)

            if sum_attention is None:
                sum_attention = np.zeros_like(attn_map)
            sum_attention += attn_map
            count += 1

            plt.figure(figsize=(12, 6))
            im = plt.imshow(attn_map, aspect='auto', interpolation='nearest', cmap='viridis', origin='lower')
            plt.colorbar(im, label='Attention Weight')
            plt.axvline(x=context_length - 0.5, color='red', linestyle='--', linewidth=2, label='Start of Prediction')
            plt.xlabel("Time Steps (Past -> Future)")
            plt.ylabel("Prediction Steps (Future)")

            if item_ids is not None and i < len(item_ids):
                safe_item_id = str(item_ids[i]).replace("/", "_").replace("\\", "_")
                title = f"Attention Map: {item_ids[i]}"
                filename = f"attention_{safe_item_id}.png"
            else:
                title = f"Attention Map (Sample {i})"
                filename = f"attention_sample_{i:04d}.png"

            plt.title(title, fontsize=12)
            plt.tight_layout()
            filepath = os.path.join(individual_dir, filename)
            plt.savefig(filepath, bbox_inches='tight', dpi=100)
            plt.close()

            if (i + 1) % 100 == 0:
                print(f"Saved {i + 1}/{len(temporal_weights_list)} attention maps")

        if sum_attention is not None:
            mean_temporal = sum_attention / count
            plt.figure(figsize=(12, 6))
            im = plt.imshow(mean_temporal, aspect='auto', interpolation='nearest', cmap='viridis', origin='lower')
            plt.colorbar(im, label='Mean Attention Weight')
            plt.axvline(x=context_length - 0.5, color='red', linestyle='--', linewidth=2, label='Start of Prediction')
            plt.xlabel("Time Steps (Past -> Future)")
            plt.ylabel("Prediction Steps (Future)")
            plt.title(f"Average Temporal Attention Matrix (N={count})", fontsize=14)
            plt.tight_layout()
            plt.savefig(os.path.join(self.save_dir, 'mean_temporal_attention.png'))
            plt.close()

        print(f"All individual attention maps saved to: {individual_dir}")


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
        static_features=static_df_formatted
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


def plot_training_history(log_dir='lightning_logs', save_path='figure/training_history.png'):
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
    estimator = TemporalFusionTransformerEstimator(
        freq="D",
        prediction_length=PREDICTION_LENGTH,
        context_length=CONTEXT_LENGTH,
        static_cardinalities=cardinality,
        distr_output=NegativeBinomialOutput(),
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


def process_logs(log_dir, save_path):
    print("\n=== Plotting Training History ===")
    plot_training_history(log_dir, save_path)
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
    print("\n=== Evaluating & Collecting Interpretability Data ===")
    collector = InterpretabilityCollector(predictor.prediction_net.model)
    collector.register()
    forecast_it, ts_it = make_evaluation_predictions(
        dataset=full_dataset,
        predictor=predictor,
        num_samples=100
    )
    forecasts = list(forecast_it)
    tss = list(ts_it)
    collector.remove()
    time_feat_names = [t.__class__.__name__ for t in time_features_from_frequency_str("D")]
    item_ids = [forecast.item_id for forecast in forecasts]
    collector.save_and_plot_summary(static_cols, time_feat_names, CONTEXT_LENGTH, item_ids)
    evaluator = Evaluator()
    agg_metrics, item_metrics = evaluator(tss, forecasts)
    print("\n=== Metrics ===")
    for key, value in agg_metrics.items():
        print(f"{key}: {value:.4f}")
    return forecasts, tss


def process_predictions(forecasts, tss, quartile_books):
    print("\n=== Saving Predictions ===")
    for i, forecast in enumerate(forecasts):
        book_id = forecast.item_id
        actual_data_log = tss[i]
        safe_book_id = book_id.replace("/", "_").replace("\\", "_")
        base_save_path = f"figure/predict/forecast_{safe_book_id}.png"
        save_forecast(
            forecast=forecast,
            actual_data_log=actual_data_log,
            save_path=base_save_path,
            title=f"{book_id}",
            use_log_scale=USE_LOG_SCALE,
            prediction_length=PREDICTION_LENGTH,
            context_length=CONTEXT_LENGTH
        )
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
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(forecasts)}")
    print(f"All {len(forecasts)} prediction plots saved to figure/predict/")


def main():
    os.makedirs('figure', exist_ok=True)
    os.makedirs('figure/predict', exist_ok=True)
    os.makedirs('figure/interpretability', exist_ok=True)

    print("=== Loading Data ===")
    df = pd.read_parquet('data/df_for.parquet')

    quartile_books = extract_quartile_books(df)
    scaling_data(df)
    full_dataset, train_dataset, cardinality = prepare_dataset(df)
    predictor = train_model(train_dataset, cardinality)
    process_logs('lightning_logs/tft_training', 'figure/training_history.png')

    forecasts, tss = evaluate_prediction(predictor, full_dataset)
    process_predictions(forecasts, tss, quartile_books)

    print("\nAll plots saved. Process Complete.")

if __name__ == "__main__":
    main()
