import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from chronos import ChronosPipeline
import os
from tqdm import tqdm

plt.rcParams['font.family'] = ['Hiragino Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

MODEL_NAME = "amazon/chronos-t5-small"
PREDICTION_LENGTH = 45
CONTEXT_LENGTH = 180
BATCH_SIZE = 16

ALL_TITLE_PREDICT = False


def extract_quartile_books(df):
    print("\n=== Identifying Representative Books (Quantiles) ===")
    total_sales = df.groupby('書名', observed=True)['POS販売冊数'].sum().sort_values()
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


def prepare_samples(df, quartile_books):
    print("\n=== Preparing Samples ===")
    grouped = df.groupby('書名', observed=True)['POS販売冊数']
    samples = []

    for name, group in grouped:
        if not ALL_TITLE_PREDICT and name not in quartile_books:
            continue

        full_values = group.values
        total_len = len(full_values)
        min_len = CONTEXT_LENGTH + PREDICTION_LENGTH

        if total_len < min_len:
            continue

        context_data = full_values[-(CONTEXT_LENGTH + PREDICTION_LENGTH):-PREDICTION_LENGTH]
        target_data = full_values[-PREDICTION_LENGTH:]

        sample = {
            "book_id": name,
            "context": torch.tensor(context_data, dtype=torch.float32),
            "target": torch.tensor(target_data, dtype=torch.float32)
        }
        samples.append(sample)

    print(f"Created {len(samples)} samples.")
    return samples


def load_model():
    print(f"\n=== Loading Pre-trained Model: {MODEL_NAME} ===")
    pipeline = ChronosPipeline.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        dtype=torch.bfloat16,
    )
    return pipeline


def run_inference(pipeline, samples):
    print("\n=== Starting Inference ===")
    context_list = [s["context"] for s in samples]
    forecasts = []

    for i in tqdm(range(0, len(context_list), BATCH_SIZE)):
        batch_context = context_list[i:i + BATCH_SIZE]
        batch_forecast = pipeline.predict(
            inputs=batch_context,
            prediction_length=PREDICTION_LENGTH,
            num_samples=100,
            limit_prediction_length=False
        )
        forecasts.extend(batch_forecast)

    return forecasts


def calculate_decile_groups(df):
    print("\n=== Calculating Decile Groups ===")
    total_sales = df.groupby('書名', observed=True)['POS販売冊数'].sum().sort_values(ascending=False)
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
        os.makedirs(f'chronos_predict/predict/{folder}', exist_ok=True)
        count = sum(1 for v in decile_groups.values() if v == decile)
        decile_sales = sum(total_sales[book] for book, d in decile_groups.items() if d == decile)
        decile_pct = (decile_sales / total_sales_all) * 100
        print(f"  {folder}: {count} books, {decile_pct:.1f}% of total sales")

    return decile_groups


def save_predictions(samples, forecasts, quartile_books, decile_groups):
    print("\n=== Saving Predictions ===")

    for i, sample in enumerate(samples):
        book_id = sample["book_id"]
        context_np = sample["context"].numpy()
        target_np = sample["target"].numpy()
        forecast_np = forecasts[i].numpy()

        plt.figure(figsize=(12, 6))

        context_x = range(0, len(context_np))
        target_x = range(len(context_np), len(context_np) + len(target_np))

        plt.plot(context_x, context_np, color="black", alpha=0.5, label="History (Context)")
        plt.plot(target_x, target_np, color="gray", linestyle="--", alpha=0.8, label="Actual Target")

        median = np.median(forecast_np, axis=0)
        low_80 = np.quantile(forecast_np, 0.1, axis=0)
        high_80 = np.quantile(forecast_np, 0.9, axis=0)

        plt.plot(target_x, median, color="blue", linewidth=2, label="Forecast (Median)")
        plt.fill_between(target_x, low_80, high_80, color="blue", alpha=0.2, label="80% Interval")

        plt.axvline(x=len(context_np), color="red", linestyle=":", alpha=0.5)

        title = f"{book_id}"
        if book_id in quartile_books:
            label = quartile_books[book_id]["label"]
            total_sales_val = quartile_books[book_id]["total_sales"]
            title = f"[{label}] {book_id}\nTotal Sales: {total_sales_val:,.0f} books"

        plt.title(title, fontsize=14)
        plt.xlabel("Time Index")
        plt.ylabel("POS販売冊数")
        plt.legend(loc='upper left')
        plt.grid(True, alpha=0.5)
        plt.tight_layout()

        safe_book_id = book_id.replace("/", "_").replace("\\", "_")

        if ALL_TITLE_PREDICT:
            decile = decile_groups.get(book_id, 10)
            if decile == 1:
                folder = "top_10"
            elif decile == 10:
                folder = "90-100"
            else:
                folder = f"{(decile-1)*10}-{decile*10}"
            save_path = f"chronos_predict/predict/{folder}/forecast_{safe_book_id}.png"
            plt.savefig(save_path)

        if book_id in quartile_books:
            info = quartile_books[book_id]
            label = info["label"]
            safe_label = label.replace(" ", "").replace("(", "").replace(")", "").replace("%", "")
            rep_save_path = f"chronos_predict/forecast_{safe_label}_{safe_book_id}.png"
            plt.savefig(rep_save_path)

        plt.close()

    if ALL_TITLE_PREDICT:
        print(f"All {len(samples)} prediction plots saved to chronos_predict/predict/*/")
    else:
        print(f"{len(samples)} representative prediction plots saved to chronos_predict/")


def main():
    print("=== Loading Data ===")
    df = pd.read_parquet('data/df_for.parquet')
    df = df.sort_values(['書名', '日付'])
    print("Complete!")

    quartile_books = extract_quartile_books(df)
    samples = prepare_samples(df, quartile_books)
    pipeline = load_model()
    forecasts = run_inference(pipeline, samples)
    decile_groups = calculate_decile_groups(df)
    save_predictions(samples, forecasts, quartile_books, decile_groups)

    print("\nAll plots saved. Process Complete.")


if __name__ == "__main__":
    main()
