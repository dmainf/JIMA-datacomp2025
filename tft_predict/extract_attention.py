import pandas as pd
import numpy as np
from gluonts.evaluation import make_evaluation_predictions
from gluonts.time_feature import time_features_from_frequency_str
import matplotlib.pyplot as plt
import os

plt.rcParams['font.family'] = ['Hiragino Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

CONTEXT_LENGTH = 180
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


def extract_attention(predictor, full_dataset, save_dir='figure/interpretability'):
    print("\n=== Extracting Attention and Interpretability Data ===")
    collector = InterpretabilityCollector(predictor.prediction_net.model, save_dir=save_dir)
    collector.register()
    forecast_it, ts_it = make_evaluation_predictions(
        dataset=full_dataset,
        predictor=predictor,
        num_samples=100
    )
    forecasts = list(forecast_it)
    list(ts_it)
    collector.remove()
    time_feat_names = [t.__class__.__name__ for t in time_features_from_frequency_str("D")]
    item_ids = [forecast.item_id for forecast in forecasts]
    collector.save_and_plot_summary(static_cols, time_feat_names, CONTEXT_LENGTH, item_ids)
    print("Attention extraction complete.")


if __name__ == "__main__":
    print("This script is designed to be imported and used with a trained predictor.")
    print("Usage example:")
    print("  from extract_attention import extract_attention")
    print("  extract_attention(predictor, full_dataset)")
