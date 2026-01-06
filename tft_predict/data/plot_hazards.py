import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='.*Glyph.*missing from font.*')
plt.rcParams['font.family'] = ['Hiragino Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


SPIKE_THRESHOLDS = [2.0, 2.5, 3.0]
CONTEXT_LENGTH = 180
RECENT_PERIOD = 14


def calculate_hazard_features(df, context_length=180, recent_period=14, spike_threshold_zscore=1.04):
    print("\n=== Calculating Hazard Features ===")
    print(f"Context length: {context_length} days")
    print(f"Recent period (momentum): {recent_period} days")
    print(f"Spike threshold: z-score >= {spike_threshold_zscore}")

    df = df.sort_values(['書名', '日付']).reset_index(drop=True)

    n = len(df)
    momentum_arr = np.ones(n, dtype=np.float32)
    volatility_arr = np.zeros(n, dtype=np.float32)
    days_since_arr = np.zeros(n, dtype=np.int32)
    z_score_arr = np.zeros(n, dtype=np.float32)

    for book_name, group in df.groupby('書名', observed=False):
        indices = group.index.values
        sales = group['POS販売冊数'].values

        last_spike_idx = -1

        for i in range(len(sales)):
            current_idx = indices[i]

            start_local = max(0, i - context_length + 1)
            hist_sales = sales[start_local : i + 1]

            if len(hist_sales) >= recent_period:
                rec_mean = np.mean(hist_sales[-recent_period:])
                hist_mean = np.mean(hist_sales)
                momentum = rec_mean / hist_mean if hist_mean > 0 else 1.0
            else:
                momentum = 1.0

            volatility = np.std(hist_sales) if len(hist_sales) > 1 else 0.0

            current_z_score = 0.0
            is_spike = False

            if len(hist_sales) > 1:
                hist_mean = np.mean(hist_sales)
                hist_std = np.std(hist_sales)
                if hist_std > 0:
                    current_z_score = (sales[i] - hist_mean) / hist_std
                    if current_z_score >= spike_threshold_zscore:
                        is_spike = True

            if is_spike:
                last_spike_idx = i
                days_since = 0
            else:
                if last_spike_idx == -1:
                    days_since = 0
                else:
                    days_since = i - last_spike_idx

            momentum_arr[current_idx] = momentum
            volatility_arr[current_idx] = volatility
            days_since_arr[current_idx] = days_since
            z_score_arr[current_idx] = current_z_score

    df['momentum'] = momentum_arr
    df['volatility'] = volatility_arr
    df['days_since_spike'] = days_since_arr
    df['z_score'] = z_score_arr

    print("\n=== Hazard Features Statistics ===")
    print(f"Momentum - Mean: {df['momentum'].mean():.4f}, Std: {df['momentum'].std():.4f}")
    print(f"Volatility - Mean: {df['volatility'].mean():.4f}, Std: {df['volatility'].std():.4f}")
    print(f"Days since spike - Mean: {df['days_since_spike'].mean():.1f}, Median: {df['days_since_spike'].median():.1f}")
    print(f"Z-score - Mean: {df['z_score'].mean():.4f}, Std: {df['z_score'].std():.4f}")

    return df


print("=== Loading Data ===")
df_original = pd.read_parquet('df_for.parquet')
print("Complete!")

for SPIKE_THRESHOLD_ZSCORE in SPIKE_THRESHOLDS:
    print(f"\n{'='*60}")
    print(f"Processing with Z-score threshold: {SPIKE_THRESHOLD_ZSCORE}")
    print(f"{'='*60}")

    df = df_original.copy()
    df = calculate_hazard_features(df, CONTEXT_LENGTH, RECENT_PERIOD, SPIKE_THRESHOLD_ZSCORE)
    df['POS販売冊数'] = np.log1p(df['POS販売冊数'])

    print("\n=== Selecting Representative Books ===")
    total_sales = df.groupby('書名', observed=False)['POS販売冊数'].sum().sort_values()
    quantiles = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    quantile_names = ["Min (0%)", "Q1 (10%)", "Q2 (20%)", "Q3 (30%)", "Q4 (40%)", "Median (50%)", "Q6 (60%)", "Q7 (70%)", "Q8 (80%)", "Q9 (90%)", "Max (100%)"]
    representative_books = {}

    num_books = len(total_sales)
    for q, name in zip(quantiles, quantile_names):
        idx = int((num_books - 1) * q)
        book_name = total_sales.index[idx]
        sales_val = total_sales.iloc[idx]
        representative_books[book_name] = {
            "label": name,
            "total_sales": sales_val
        }
        print(f"{name:<15} | {sales_val:,.0f}冊 | {book_name}")

    threshold_dir = f'hazard/zscore_{SPIKE_THRESHOLD_ZSCORE}'
    os.makedirs(threshold_dir, exist_ok=True)

    print("\n=== Plotting Hazard Features ===")
    for book_name, info in representative_books.items():
        label = info["label"]
        total_sales_val = info["total_sales"]

        book_df = df[df['書名'] == book_name].sort_values('日付').reset_index(drop=True)

        fig, axes = plt.subplots(5, 1, figsize=(14, 12), sharex=True)

        axes[0].plot(book_df['日付'], book_df['POS販売冊数'], color='black', linewidth=1)
        axes[0].set_ylabel('POS販売冊数', fontsize=11)
        axes[0].set_title(f'[{label}] {book_name}\nTotal Sales: {total_sales_val:,.0f} books (Z-score threshold: {SPIKE_THRESHOLD_ZSCORE})',
                          fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(book_df['日付'], book_df['momentum'], color='blue', linewidth=1)
        axes[1].axhline(y=1.0, color='red', linestyle='--', linewidth=0.8, alpha=0.5)
        axes[1].set_ylabel('Momentum', fontsize=11)
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(book_df['日付'], book_df['volatility'], color='green', linewidth=1)
        axes[2].set_ylabel('Volatility', fontsize=11)
        axes[2].grid(True, alpha=0.3)

        axes[3].plot(book_df['日付'], book_df['days_since_spike'], color='orange', linewidth=1)
        axes[3].set_ylabel('Days Since Spike', fontsize=11)
        axes[3].grid(True, alpha=0.3)

        axes[4].plot(book_df['日付'], book_df['z_score'], color='purple', linewidth=1)
        axes[4].axhline(y=SPIKE_THRESHOLD_ZSCORE, color='red', linestyle='--', linewidth=0.8, alpha=0.5, label=f'Spike Threshold ({SPIKE_THRESHOLD_ZSCORE})')
        axes[4].set_ylabel('Z-score', fontsize=11)
        axes[4].set_xlabel('日付', fontsize=11)
        axes[4].grid(True, alpha=0.3)
        axes[4].legend(fontsize=9)

        plt.tight_layout()

        safe_label = label.replace(" ", "").replace("(", "").replace(")", "").replace("%", "")
        safe_book_name = book_name.replace("/", "_").replace("\\", "_")
        save_path = f'{threshold_dir}/hazard_{safe_label}_{safe_book_name}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Saved: {save_path}")

    print("\n=== Creating Summary Statistics Plot ===")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    axes[0, 0].hist(df['POS販売冊数'], bins=100, color='black', alpha=0.7, edgecolor='black')
    axes[0, 0].set_xlabel('POS販売冊数', fontsize=11)
    axes[0, 0].set_ylabel('Frequency', fontsize=11)
    axes[0, 0].set_title('Distribution of POS Sales', fontsize=12, fontweight='bold')
    axes[0, 0].set_yscale('log')
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].hist(df['momentum'], bins=100, color='blue', alpha=0.7, edgecolor='black')
    axes[0, 1].set_xlabel('Momentum', fontsize=11)
    axes[0, 1].set_ylabel('Frequency', fontsize=11)
    axes[0, 1].set_title(f'Distribution of Momentum\nMean: {df["momentum"].mean():.4f}, Std: {df["momentum"].std():.4f}',
                         fontsize=12, fontweight='bold')
    axes[0, 1].set_yscale('log')
    axes[0, 1].axvline(x=1.0, color='red', linestyle='--', linewidth=1.5)
    axes[0, 1].grid(True, alpha=0.3)

    axes[0, 2].hist(df['volatility'], bins=100, color='green', alpha=0.7, edgecolor='black')
    axes[0, 2].set_xlabel('Volatility', fontsize=11)
    axes[0, 2].set_ylabel('Frequency', fontsize=11)
    axes[0, 2].set_title(f'Distribution of Volatility\nMean: {df["volatility"].mean():.4f}, Std: {df["volatility"].std():.4f}',
                         fontsize=12, fontweight='bold')
    axes[0, 2].set_yscale('log')
    axes[0, 2].grid(True, alpha=0.3)

    axes[1, 0].hist(df['days_since_spike'], bins=100, color='orange', alpha=0.7, edgecolor='black')
    axes[1, 0].set_xlabel('Days Since Spike', fontsize=11)
    axes[1, 0].set_ylabel('Frequency', fontsize=11)
    axes[1, 0].set_title(f'Distribution of Days Since Spike\nMean: {df["days_since_spike"].mean():.1f}, Median: {df["days_since_spike"].median():.1f}',
                         fontsize=12, fontweight='bold')
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].hist(df['z_score'], bins=100, color='purple', alpha=0.7, edgecolor='black')
    axes[1, 1].set_xlabel('Z-score', fontsize=11)
    axes[1, 1].set_ylabel('Frequency', fontsize=11)
    axes[1, 1].set_title(f'Distribution of Z-score\nMean: {df["z_score"].mean():.4f}, Std: {df["z_score"].std():.4f}',
                         fontsize=12, fontweight='bold')
    axes[1, 1].set_yscale('log')
    axes[1, 1].axvline(x=SPIKE_THRESHOLD_ZSCORE, color='red', linestyle='--', linewidth=1.5, label=f'Spike Threshold ({SPIKE_THRESHOLD_ZSCORE})')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend(fontsize=9)

    axes[1, 2].axis('off')

    plt.tight_layout()
    summary_path = f'{threshold_dir}/hazard_summary_statistics.png'
    plt.savefig(summary_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {summary_path}")

print("\nAll plots saved to hazard/")
