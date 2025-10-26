import pandas as pd
import numpy as np
from pathlib import Path
from multiprocessing import Pool, cpu_count
from sklearn.preprocessing import LabelEncoder
from common import BASE_DATE

TEMPORAL_FEATURE_COLS = ['day_of_week', 'is_weekend', 'month', 'day_of_month', 'quarter']
LAG_WINDOWS = [1, 3, 7, 14, 28]
ROLL_WINDOWS = [3, 7, 14, 28]

def get_feature_dimensions():
    num_dynamic_real = len(TEMPORAL_FEATURE_COLS)
    num_lag = len(LAG_WINDOWS)
    num_roll = len(ROLL_WINDOWS) * 3
    num_past_dynamic = num_lag + num_roll
    return num_dynamic_real, num_past_dynamic

def label_enc(df, columns):
    df_encoded = df.copy()
    encoders = {}
    for col in columns:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        encoders[col] = le
    return df_encoded, encoders

def prepare_dataset(df, label_columns, context_length, prediction_length, sales_percentile):
    dataset = df.copy()
    dataset['day_of_week'] = ((dataset['累積日数'] - 1) % 7).astype(np.int8)
    dataset['is_weekend'] = (dataset['day_of_week'] >= 5).astype(np.int8)
    dataset['week_of_year'] = ((dataset['累積日数'] - 1) // 7).astype(np.int16)
    dates = pd.to_datetime(BASE_DATE + pd.to_timedelta(dataset['累積日数'] - 1, unit='D'))
    dataset['month'] = dates.dt.month.astype(np.int8)
    dataset['day_of_month'] = dates.dt.day.astype(np.int8)
    dataset['quarter'] = dates.dt.quarter.astype(np.int8)

    group_stats = dataset.groupby(['書店コード', '書名'])['累積日数'].agg(['count', 'min', 'max'])
    min_length = context_length + prediction_length
    valid_groups = group_stats[group_stats['count'] >= min_length].index
    dataset = dataset.set_index(['書店コード', '書名']).loc[valid_groups].reset_index()

    book_sales = dataset.groupby('書名')['売上'].sum()
    threshold = book_sales.quantile(sales_percentile)
    high_sales_books = book_sales[book_sales >= threshold].index
    dataset = dataset[dataset['書名'].isin(high_sales_books)]
    dataset = dataset.sort_values(['書店コード', '書名', '累積日数']).reset_index(drop=True)

    dataset, encoders = label_enc(dataset, label_columns)

    dataset = dataset.sort_values(['書店コード', '書名', '累積日数']).copy()
    for lag in LAG_WINDOWS:
        dataset[f'売上_lag{lag}'] = dataset.groupby(['書店コード', '書名'])['売上'].shift(lag)
    for window in ROLL_WINDOWS:
        dataset[f'売上_roll_mean{window}'] = dataset.groupby(['書店コード', '書名'])['売上'].rolling(window, min_periods=1).mean().reset_index(level=[0,1], drop=True)
        dataset[f'売上_roll_std{window}'] = dataset.groupby(['書店コード', '書名'])['売上'].rolling(window, min_periods=1).std().reset_index(level=[0,1], drop=True).fillna(0)
        dataset[f'売上_roll_max{window}'] = dataset.groupby(['書店コード', '書名'])['売上'].rolling(window, min_periods=1).max().reset_index(level=[0,1], drop=True)
    dataset = dataset.fillna(0)

    top_pct_high = 0.8
    top_pct_mid = 0.5
    book_total_sales = dataset.groupby(['書店コード', '書名'])['売上'].transform('sum')
    unique_sales = book_total_sales.unique()
    if len(unique_sales) > 0:
        threshold_high = np.percentile(unique_sales, top_pct_high * 100)
        threshold_mid = np.percentile(unique_sales, top_pct_mid * 100)
        conditions = [
            book_total_sales >= threshold_high,
            book_total_sales >= threshold_mid
        ]
        choices = [
            3.0,
            2.0
        ]
        weights = np.select(conditions, choices, default=1.0)
    else:
        weights = 1.0
    dataset['sample_weight'] = weights

    print(f"Valid books after filtering: {len(high_sales_books)}")
    print(f"Total records: {len(dataset)}")
    return dataset, encoders


def create_gluonts_dataset(dataset, context_length, prediction_length, num_samples_per_series=3, sample_length_multiplier=2.5, use_bidirectional=False):
    num_cores = max(1, cpu_count() - 1)
    train_groups_args = []
    test_groups_args = []
    min_length = context_length + prediction_length
    target_sample_length = int(min_length * sample_length_multiplier)
    for (store, book), group in dataset.groupby(['書店コード', '書名'], sort=False):
        group_sorted = group.sort_values('累積日数')
        total_len = len(group_sorted)
        #longer data, can divide more sanple
        if total_len < min_length:
            continue
        if total_len < min_length * 1.5:
            n_samples = 1
        elif total_len < min_length * 3:
            n_samples = 2
        else:
            n_samples = num_samples_per_series
        sample_len = min(target_sample_length, total_len)
        max_start = total_len - sample_len

        if max_start <= 0:
            train_groups_args.append(((store, book), group_sorted))
        else:
            if n_samples == 1:
                start_idx = np.random.randint(0, max_start + 1)
                sample = group_sorted.iloc[start_idx:start_idx + sample_len]
                train_groups_args.append(((store, book), sample))
            else:
                region_size = max_start / n_samples
                for i in range(n_samples):
                    region_start = int(i * region_size)
                    region_end = int((i + 1) * region_size) if i < n_samples - 1 else max_start + 1

                    if region_end <= region_start:
                        region_end = region_start + 1

                    if region_end > max_start + 1:
                        region_end = max_start + 1

                    start_idx = np.random.randint(region_start, region_end)
                    sample = group_sorted.iloc[start_idx:start_idx + sample_len]
                    train_groups_args.append(((store, book), sample))
                    if use_bidirectional:
                        reversed_sample = sample.iloc[::-1].copy()
                        train_groups_args.append(((store, book), reversed_sample))

        last_possible = total_len - prediction_length
        first_possible = context_length
        if last_possible > first_possible:
            eval_point = np.random.randint(first_possible, last_possible + 1)
            eval_sample = group_sorted.iloc[:eval_point + prediction_length]
            test_groups_args.append(((store, book), eval_sample))
        else:
            test_groups_args.append(((store, book), group_sorted))

    with Pool(num_cores) as pool:
        print(f"Processing {len(train_groups_args)} training samples...")
        train_list = pool.map(process_group, train_groups_args)
        print(f"Processing {len(test_groups_args)} test samples...")
        test_list = pool.map(process_group, test_groups_args)

    return train_list, test_list


def process_group(args):
    _, group = args
    group_sorted = group.sort_values('累積日数')
    static_cat = [
        int(group_sorted['書店コード'].iloc[0]),
        int(group_sorted['出版社'].iloc[0]),
        int(group_sorted['著者名'].iloc[0]),
        int(group_sorted['大分類'].iloc[0]),
        int(group_sorted['中分類'].iloc[0]),
        int(group_sorted['小分類'].iloc[0])
    ]
    static_real = [
        float(group_sorted['本体価格'].iloc[0]),
        float(group_sorted['sample_weight'].iloc[0])
    ]
    start_day = int(group_sorted['累積日数'].iloc[0])
    start_date = BASE_DATE + pd.Timedelta(days=start_day - 1)

    feat_dynamic_real = np.stack([
        group_sorted[col].values for col in TEMPORAL_FEATURE_COLS
    ], axis=0).astype(np.float32)

    lag_cols = [f'売上_lag{lag}' for lag in LAG_WINDOWS]
    roll_mean_cols = [f'売上_roll_mean{w}' for w in ROLL_WINDOWS]
    roll_std_cols = [f'売上_roll_std{w}' for w in ROLL_WINDOWS]
    roll_max_cols = [f'売上_roll_max{w}' for w in ROLL_WINDOWS]

    past_dynamic_cols = lag_cols + roll_mean_cols + roll_std_cols + roll_max_cols
    feat_dynamic_past = group_sorted[past_dynamic_cols].values.T.astype(np.float32)

    return {
        "start": pd.Period(start_date, freq="D"),
        "target": group_sorted['売上'].values.astype(np.float32),
        "feat_static_cat": static_cat,
        "feat_static_real": static_real,
        "feat_dynamic_real": feat_dynamic_real,
        "past_feat_dynamic_real": feat_dynamic_past
    }
