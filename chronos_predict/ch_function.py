import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from chronos import ChronosPipeline
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
from accelerate import Accelerator
from torch.utils.data import Dataset, random_split
import os
from tqdm import tqdm
import warnings
import logging

warnings.filterwarnings("ignore")
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

plt.rcParams['font.family'] = ['Hiragino Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

CONFIG = {
    "model_name": "amazon/chronos-t5-small",
    "prediction_length": 64,
    "context_length": 180,
    "batch_size": 16,
    "use_log_scale": False,
    "num_samples": 20,
    "output_dir": "ch_valid",
    "lora_output_dir": "ch_lora_checkpoints",
    "learning_rate": 1e-4,
    "epochs": 3
}

class ChronosDataset(Dataset):
    def __init__(self, df, tokenizer, context_length, prediction_length, use_log_scale=True, mode="train"):
        self.samples = []
        self.tokenizer = tokenizer

        grouped = df.groupby('書名', observed=True)
        for _, group in grouped:
            if len(group) < context_length + prediction_length:
                continue

            group = group.sort_values('日付')
            raw_sales = group['POS販売冊数'].values.astype(np.float32)

            if use_log_scale:
                series_data = np.log1p(raw_sales)
            else:
                series_data = raw_sales

            total_len = context_length + prediction_length

            if mode == "train":
                valid_end_idx = len(series_data) - prediction_length
            else:
                valid_end_idx = len(series_data)

            max_start_idx = valid_end_idx - total_len + 1

            if max_start_idx <= 0:
                continue

            for i in range(max_start_idx):
                window = series_data[i : i + total_len]
                context = window[:context_length]
                target = window[context_length:]

                self.samples.append((torch.tensor(context), torch.tensor(target)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        context, target = self.samples[idx]

        input_ids, attention_mask, scale = self.tokenizer.context_input_transform(context.unsqueeze(0))
        labels, _ = self.tokenizer.label_input_transform(target.unsqueeze(0), scale)

        return {
            "input_ids": input_ids.squeeze(0),
            "attention_mask": attention_mask.squeeze(0),
            "labels": labels.squeeze(0)
        }

def extract_decile_books(df):
    print("\n=== Identifying Representative Books (5% intervals) ===")
    total_sales = df.groupby('書名')['POS販売冊数'].sum().sort_values()

    quantiles = [i / 20 for i in range(21)]

    decile_books = {}
    print(f"{'Percentile':<15} | {'Total Sales':<12} | {'Book Name'}")
    print("-" * 60)

    num_books = len(total_sales)

    for q in quantiles:
        percent = int(round(q * 100))
        idx = int((num_books - 1) * q)
        book_name = total_sales.index[idx]
        sales_val = total_sales.iloc[idx]

        label_name = f"{percent}%"
        file_prefix = f"{percent}%"

        decile_books[book_name] = {
            "label": label_name,
            "total_sales": sales_val,
            "file_prefix": file_prefix
        }
        print(f"{label_name:<15} | {sales_val:,.0f}冊       | {book_name}")

    return decile_books

def train_model(df):
    print("\n=== Starting LoRA Fine-Tuning (Baseline) ===")

    pipeline = ChronosPipeline.from_pretrained(
        CONFIG["model_name"],
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
    )

    tokenizer = pipeline.tokenizer
    model = pipeline.inner_model

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        inference_mode=False,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q", "k", "v", "o", "wi", "wo"]
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    full_dataset = ChronosDataset(
        df,
        tokenizer,
        context_length=CONFIG["context_length"],
        prediction_length=CONFIG["prediction_length"],
        use_log_scale=CONFIG["use_log_scale"],
        mode="train"
    )

    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    print(f"Total samples: {len(full_dataset)} | Train: {len(train_dataset)} | Val: {len(val_dataset)}")

    training_args = Seq2SeqTrainingArguments(
        output_dir=CONFIG["lora_output_dir"],
        per_device_train_batch_size=CONFIG["batch_size"],
        per_device_eval_batch_size=CONFIG["batch_size"],
        learning_rate=CONFIG["learning_rate"],
        num_train_epochs=CONFIG["epochs"],
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        remove_unused_columns=False,
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss"
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=None
    )

    trainer.train()

    final_adapter_path = os.path.join(CONFIG["lora_output_dir"], "final_adapter")
    model.save_pretrained(final_adapter_path)
    print(f"LoRA adapter saved to: {final_adapter_path}")
    return final_adapter_path

def load_model(adapter_path=None):
    accelerator = Accelerator()

    if torch.cuda.is_available():
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float32

    base_pipeline = ChronosPipeline.from_pretrained(
        CONFIG["model_name"],
        device_map="auto",
        torch_dtype=torch_dtype
    )

    if adapter_path and os.path.exists(adapter_path):
        print(f"Loading LoRA adapter from: {adapter_path}")
        inner_model = PeftModel.from_pretrained(base_pipeline.inner_model, adapter_path)
        base_pipeline.model.model = inner_model
        base_pipeline.inner_model = inner_model
    else:
        print("Using base model.")

    print(f"Using device: {accelerator.device}")
    return base_pipeline, accelerator

def preprocess_data(df, decile_books=None, all_predict=True):
    samples = []
    grouped = df.groupby('書名', observed=True)

    for book_name, group in grouped:
        if not all_predict and decile_books is not None and book_name not in decile_books:
            continue

        if len(group) < CONFIG["context_length"] + CONFIG["prediction_length"]:
            continue

        group = group.sort_values('日付')
        raw_sales = group['POS販売冊数'].values.astype(np.float32)

        if CONFIG["use_log_scale"]:
            series_data = np.log1p(raw_sales)
        else:
            series_data = raw_sales

        total_len = CONFIG["context_length"] + CONFIG["prediction_length"]
        context = series_data[-total_len:-CONFIG["prediction_length"]]
        target = series_data[-CONFIG["prediction_length"]:]

        samples.append({
            "id": book_name,
            "context": torch.tensor(context, dtype=torch.float32),
            "target": torch.tensor(target, dtype=torch.float32),
            "raw_target": raw_sales[-CONFIG["prediction_length"]:]
        })

    return samples

def run_inference(pipeline, samples, accelerator):
    forecasts = []
    context_list = [s["context"] for s in samples]

    for i in tqdm(range(0, len(context_list), CONFIG["batch_size"])):
        batch_context = context_list[i:i + CONFIG["batch_size"]]

        batch_forecast = pipeline.predict(
            batch_context,
            prediction_length=CONFIG["prediction_length"],
            num_samples=CONFIG["num_samples"],
            limit_prediction_length=False
        )
        forecasts.extend(batch_forecast)

    return forecasts

def save_results(samples, forecasts, decile_books, all_predict):
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    if all_predict:
        os.makedirs("ch_valid/All_predict", exist_ok=True)
        print("\n=== Saving decile_books + ALL book predictions ===")
    else:
        print("\n=== Saving representative book predictions (decile_books) ===")

    saved_count = 0
    for i, sample in enumerate(samples):
        book_name = sample['id']

        forecast = forecasts[i].numpy()
        context = sample["context"].numpy()
        target = sample["target"].numpy()

        if CONFIG["use_log_scale"]:
            forecast = np.expm1(forecast)
            context = np.expm1(context)
            target = np.expm1(target)

        plt.figure(figsize=(10, 6))

        context_idx = np.arange(len(context))
        target_idx = np.arange(len(context), len(context) + len(target))

        plt.plot(context_idx, context, color="black", alpha=0.5, label="History")
        plt.plot(target_idx, target, color="gray", linestyle="--", label="Actual")

        median = np.median(forecast, axis=0)
        low_10 = np.quantile(forecast, 0.1, axis=0)
        high_90 = np.quantile(forecast, 0.9, axis=0)

        plt.plot(target_idx, median, color="blue", label="Median Forecast")
        plt.fill_between(target_idx, low_10, high_90, color="blue", alpha=0.2, label="80% CI")

        if book_name in decile_books:
            info = decile_books[book_name]
            label = info["label"]
            total_sales = info["total_sales"]
            plt.title(f"[{label}] {book_name}\nTotal Sales: {total_sales:,.0f} books")
        else:
            plt.title(f"Forecast: {book_name}")

        plt.legend()
        plt.grid(True, alpha=0.3)

        safe_name = str(book_name).replace("/", "_").replace("\\", "_")

        if book_name in decile_books:
            prefix = decile_books[book_name]["file_prefix"]
            filename = f"{prefix}_{safe_name}.png"
            save_path = f"{CONFIG['output_dir']}/{filename}"
            plt.savefig(save_path)
            saved_count += 1

        if all_predict:
            filename = f"{safe_name}.png"
            save_path = f"ch_valid/All_predict/{filename}"
            plt.savefig(save_path)
            saved_count += 1

        plt.close()

        if (saved_count) % 100 == 0:
            print(f"Saved {saved_count} predictions...")

    print(f"Total saved: {saved_count} predictions")
