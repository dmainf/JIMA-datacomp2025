import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.dates as mdates
from gluonts.model.predictor import Predictor
from gluonts.dataset.common import ListDataset
from common import BASE_DATE, load_encoders
from fueture_eng import TEMPORAL_FEATURE_COLS, LAG_WINDOWS, ROLL_WINDOWS
from transformer import PREDICTION_LENGTH, CONTEXT_LENGTH

matplotlib.rcParams['font.sans-serif'] = ['Hiragino Sans', 'YuGothic', 'Hiragino Maru Gothic Pro']
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['axes.unicode_minus'] = False
print("Loading model and dataset...")
predictor = Predictor.deserialize(Path("transformer_model"))
dataset_df = pd.read_parquet('data/dataset.parquet')
encoders = load_encoders('data/encoders.pkl')
print("Loaded!")
store_codes = list(range(1, 26)) + list(range(28, 36))
output_dir = Path("score/predictions_by_store")
output_dir.mkdir(exist_ok=True, parents=True)
for store_code in store_codes:
    print(f"\nProcessing Store {store_code}...")
    store_file = Path(f"data/by_store/df_{store_code}.parquet")
    if not store_file.exists():
        print(f"  Skipped: File not found")
        continue
    df_store = pd.read_parquet(store_file)
    sales_by_book = df_store.groupby('書名')['売上'].agg(['sum', 'count'])
    sales_by_book = sales_by_book[sales_by_book['count'] >= CONTEXT_LENGTH + PREDICTION_LENGTH]
    sales_by_book = sales_by_book.sort_values('sum', ascending=False)
    if len(sales_by_book) < 3:
        print(f"  Skipped: Insufficient books")
        continue
    top_books = sales_by_book.head(5).index.tolist()
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    for idx, book_name in enumerate(top_books):
        if idx >= 5:
            break
        ax = axes[idx]
        df_book = df_store[df_store['書名'] == book_name].sort_values('累積日数')
        dates = [BASE_DATE + pd.Timedelta(days=int(d)-1) for d in df_book['累積日数'].values]
        sales = df_book['売上'].values
        # Randomize prediction interval for each book
        # Ensure minimum context and enough data for prediction
        min_split = CONTEXT_LENGTH
        max_split = len(sales) - PREDICTION_LENGTH
        if max_split <= min_split:
            # Not enough data, skip
            print(f"  Skipped {book_name[:30]}: insufficient data length")
            continue
        split_point = np.random.randint(min_split, max_split + 1)
        test_start = split_point
        test_end = min(split_point + PREDICTION_LENGTH, len(sales))
        train_dates = dates[:split_point]
        train_sales = sales[:split_point]
        test_dates = dates[test_start:test_end]
        test_sales = sales[test_start:test_end]
        ax.plot(train_dates, train_sales, color='#2C3E50', linewidth=1.0,
                label='履歴', marker='o', markersize=1.0, alpha=0.7)
        ax.plot(test_dates, test_sales, color='#E74C3C', linewidth=1.2,
                label='実測', marker='s', markersize=3)
        # Plot remaining data after prediction period
        if test_end < len(sales):
            future_dates = dates[test_end:]
            future_sales = sales[test_end:]
            ax.plot(future_dates, future_sales, color='#95A5A6', linewidth=1.0,
                    label='後続データ', marker='o', markersize=1.0, alpha=0.5)
        try:
            # Encode store_code using the same encoder as training
            if str(store_code) not in encoders['書店コード'].classes_:
                raise ValueError(f"Store {store_code} not in training data")

            encoded_store_code = encoders['書店コード'].transform([str(store_code)])[0]

            book_features = dataset_df[(dataset_df['書店コード'] == encoded_store_code) &
                                      (dataset_df['書名'] == book_name)].sort_values('累積日数')
            if len(book_features) == 0:
                raise ValueError(f"Book not in training data for store {store_code}")

            start_day = int(book_features['累積日数'].iloc[0])
            start_date = BASE_DATE + pd.Timedelta(days=start_day - 1)
            static_cat = [
                int(book_features['書店コード'].iloc[0]),
                int(book_features['出版社'].iloc[0]),
                int(book_features['著者名'].iloc[0]),
                int(book_features['大分類'].iloc[0]),
                int(book_features['中分類'].iloc[0]),
                int(book_features['小分類'].iloc[0])
            ]
            static_real = [
                float(book_features['本体価格'].iloc[0]),
                float(book_features['sample_weight'].iloc[0])
            ]

            feat_dynamic_real = book_features[TEMPORAL_FEATURE_COLS].values[:test_end].T.astype(np.float32)

            lag_cols = [f'売上_lag{lag}' for lag in LAG_WINDOWS]
            roll_mean_cols = [f'売上_roll_mean{w}' for w in ROLL_WINDOWS]
            roll_std_cols = [f'売上_roll_std{w}' for w in ROLL_WINDOWS]
            roll_max_cols = [f'売上_roll_max{w}' for w in ROLL_WINDOWS]
            past_dynamic_cols = lag_cols + roll_mean_cols + roll_std_cols + roll_max_cols
            feat_dynamic_past = book_features[past_dynamic_cols].values[:split_point].T.astype(np.float32)

            test_entry = {
                "start": pd.Period(start_date, freq="D"),
                "target": sales[:split_point],
                "feat_static_cat": static_cat,
                "feat_static_real": static_real,
                "feat_dynamic_real": feat_dynamic_real,
                "past_feat_dynamic_real": feat_dynamic_past
            }
            test_dataset = ListDataset([test_entry], freq="D")
            forecast_it = predictor.predict(test_dataset)
            forecast = list(forecast_it)[0]
            pred_median = forecast.median
            # Match prediction length with test data length
            pred_len = min(len(pred_median), len(test_dates))
            ax.plot(test_dates[:pred_len], pred_median[:pred_len], color='#3498DB', linewidth=1.2,
                   label='予測', marker='D', markersize=3, linestyle='--')
            if hasattr(forecast, 'quantile'):
                lower = forecast.quantile(0.1)
                upper = forecast.quantile(0.9)
                ax.fill_between(test_dates[:pred_len], lower[:pred_len], upper[:pred_len],
                               color='#3498DB', alpha=0.2, label='80%信頼区間')
            mae = np.mean(np.abs(test_sales[:pred_len] - pred_median[:pred_len]))
            mape = np.mean(np.abs((test_sales[:pred_len] - pred_median[:pred_len]) / (test_sales[:pred_len] + 1e-6))) * 100
            ax.text(0.02, 0.98, f'MAE: ¥{mae:.0f}\nMAPE: {mape:.1f}%',
                   transform=ax.transAxes, fontsize=8, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        except Exception as e:
            print(f"  Error: {book_name[:30]}: {str(e)[:50]}")
        ax.set_title(f'{book_name[:50]}', fontsize=9, fontweight='bold')
        ax.set_xlabel('日付', fontsize=8)
        ax.set_ylabel('売上高（円）', fontsize=8)
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.tick_params(axis='x', rotation=45, labelsize=7)
        ax.tick_params(axis='y', labelsize=7)
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m月'))
    for idx in range(len(top_books), len(axes)):
        axes[idx].axis('off')
    plt.suptitle(f'書店 {store_code} - 売上予測 Top 5 (ランダム期間、{PREDICTION_LENGTH}日間)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    output_file = output_dir / f'store_{store_code}_predictions.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    plt.close()
print(f"\nCompleted! Check {output_dir}/")
