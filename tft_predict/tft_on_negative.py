import pandas as pd
import warnings
import logging
from function import *

warnings.filterwarnings("ignore")
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
logging.getLogger("gluonts").setLevel(logging.ERROR)

PREDICTION_LENGTH = 14
CONTEXT_LENGTH = 180
EPOC = 200

ALL_TITLE_PREDICT = True
QUANTILES = [0.1, 0.5, 0.75, 0.9, 0.95, 0.99]
USE_LOG_SCALE = True

static_cols = ['出版社', '著者名', '大分類', '中分類', '小分類', '書名_base']

time_feature_cols = [
    'month_sin',
    'month_cos',
    'day_sin',
    'day_cos',
    'is_holiday'
]

relative_feature_cols = [
    '大分類_POS販売冊数_relative',
    '大分類_POS販売冊数_z_score',
    '中分類_POS販売冊数_relative',
    '中分類_POS販売冊数_z_score',
    '小分類_POS販売冊数_relative',
    '小分類_POS販売冊数_z_score',
    '大分類_log_本体価格_relative',
    '大分類_log_本体価格_z_score',
    '中分類_log_本体価格_relative',
    '中分類_log_本体価格_z_score',
    '小分類_log_本体価格_relative',
    '小分類_log_本体価格_z_score',
]

hazard_feature_cols = [
    'momentum',
    'volatility',
    'z_score',
    'days_since_spike_2.0',
    'days_since_spike_2.5',
    'days_since_spike_3.0',
    'feat_adi',
    'feat_cv2',
    'feat_hawkes',
    'feat_days_since',
    'score_sparse',
    'score_periodic',
    'score_burst',
]

past_dynamic_cols = relative_feature_cols + hazard_feature_cols


def main():
    print("=== Loading Data ===")
    df = pd.read_parquet('data/df_for.parquet')
    print("Complete!")

    decile_books = extract_decile_books(df)
    df = scaling_data(df, USE_LOG_SCALE)
    df = calculate_hazard_features(df, context_length=CONTEXT_LENGTH)
    df = calculate_temporal_relative_features(
        df,
        target_col='POS販売冊数',
        category_cols=['大分類', '中分類', '小分類'],
        context_length=CONTEXT_LENGTH
    )
    df = calculate_static_relative_features(
        df,
        target_col='log_本体価格',
        category_cols=['大分類', '中分類', '小分類']
    )
    verify_all_features(df, static_cols, time_feature_cols, relative_feature_cols, hazard_feature_cols)
    save_training_data(df, 'train.parquet')
    df.describe().to_csv("df_describe.csv")
    print("=== Preparing Dataset and Training Model ===")
    full_dataset, train_dataset, cardinality = prepare_dataset(df, static_cols, time_feature_cols, past_dynamic_cols, PREDICTION_LENGTH)
    predictor = train_model(
        train_dataset,
        cardinality,
        PREDICTION_LENGTH,
        CONTEXT_LENGTH,
        EPOC,
        QUANTILES
    )
    process_logs()
    forecasts, tss = evaluate_prediction(predictor, full_dataset)
    process_predictions(
        forecasts,
        tss,
        decile_books,
        df,
        ALL_TITLE_PREDICT,
        USE_LOG_SCALE,
        PREDICTION_LENGTH,
        CONTEXT_LENGTH,
        QUANTILES
    )
    print("\nAll plots saved. Process Complete.")
if __name__ == "__main__":
    main()
