import pandas as pd
import numpy as np
from pathlib import Path
from fueture_eng import *
from common import *
from gluonts.dataset.common import ListDataset
from gluonts.torch.model.tft import TemporalFusionTransformerEstimator
from gluonts.evaluation import make_evaluation_predictions, Evaluator
import torch

CONTEXT_LENGTH = 56
PREDICTION_LENGTH = 28

if __name__ == '__main__':
    device_type = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"\n=== Device: {device_type} ===")
    print("\n=== Loading All Store Data ===")
    df = load_all_stores(data='by_store', exclude_stores=[26, 27])
    print(f"Total records: {len(df)}")

    print("\n=== Preparing Dataset ===")
    label_columns = ['書店コード', '出版社', '著者名', '大分類', '中分類', '小分類']
    NUM_EPOCHS = 60
    BATCH_SIZE = 64
    sales_percentile = 0.2
    dataset, encoders = prepare_dataset(
        df,
        label_columns,
        CONTEXT_LENGTH,
        PREDICTION_LENGTH,
        sales_percentile
    )
    print(f"\nDataset Shape: {dataset.shape}")
    dataset.to_parquet('data/dataset.parquet')
    print("Saved dataset to data/dataset.parquet")
    save_encoders(encoders, 'data/encoders.pkl')
    print("Saved encoders to data/encoders.pkl")

    print("\n=== Creating GluonTS Dataset ===")
    train_list, test_list = create_gluonts_dataset(dataset, CONTEXT_LENGTH, PREDICTION_LENGTH)
    print(f"Train series: {len(train_list)}")
    print(f"Test series: {len(test_list)}")

    NUM_DYNAMIC_REAL_FEATURES, NUM_PAST_DYNAMIC_FEATURES = get_feature_dimensions()
    LEARNING_RATE = 3e-4
    CARDINALITY = [dataset[col].nunique() for col in label_columns]
    print(f"Feature dimensions: dynamic_real={NUM_DYNAMIC_REAL_FEATURES}, past_dynamic={NUM_PAST_DYNAMIC_FEATURES}")
    estimator = TemporalFusionTransformerEstimator(
        prediction_length=PREDICTION_LENGTH,
        freq='D',
        context_length=CONTEXT_LENGTH,
        static_cardinalities=CARDINALITY,
        static_dims=[2],
        dynamic_dims=[NUM_DYNAMIC_REAL_FEATURES],
        past_dynamic_dims=[NUM_PAST_DYNAMIC_FEATURES],
        hidden_dim=128,
        variable_dim=32,
        num_heads=4,
        dropout_rate=0.15,
        lr=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        ###8epocで改善しなかったら強制終了
        patience=8,
        ###
        trainer_kwargs={
            "max_epochs": NUM_EPOCHS,
            "accelerator": device_type,
            "devices": 1,
            "gradient_clip_val": 0.5,
        }
    )
    train_dataset = ListDataset(train_list, freq="D")
    test_dataset = ListDataset(test_list, freq="D")
    print(f"\n=== Training ===")
    predictor = estimator.train(
        train_dataset,
        validation_data=test_dataset
    )

    print(f"\n=== Evaluation ===")
    forecast_it, ts_it = make_evaluation_predictions(
        dataset=test_dataset,
        predictor=predictor,
        num_samples=100
    )
    forecasts = list(forecast_it)
    tss = list(ts_it)
    evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])
    agg_metrics, item_metrics = evaluator(tss, forecasts)

    print("\n" + "="*60)
    print("Performance Metrics")
    print("="*60)
    """
    MAE(Mean Absolute Error)
    RMSE(Root Mean Square Error)
    MAPE(Mean Absolute Percentage Error)    誤差/正解値
    sMAPE                                   誤差/(予測+正解値)
    MASE (Mean Absolute Scaled Error)       誤差/1
    """
    for key in ['MASE', 'RMSE', 'MAE', 'MAPE', 'sMAPE']:
        if key in agg_metrics:
            print(f"{key:10s}: {agg_metrics[key]:.4f}")
    mase_value = agg_metrics.get('MASE', float('inf'))
    agg_metrics_df = pd.DataFrame([agg_metrics])
    if mase_value < 0.5:
        rating = "Excellent"
    elif mase_value < 1.0:
        rating = "Good"
    elif mase_value < 1.5:
        rating = "Fair"
    else:
        rating = "Needs Improvement"
    print(f"\nRating: {rating} (MASE={mase_value:.4f})")
    if 'MAPE' in agg_metrics:
        print(f"Error: {agg_metrics['MAPE']:.2f}%")

    score_dir = Path("score")
    score_dir.mkdir(exist_ok=True)
    agg_metrics_df.to_csv(score_dir / 'transformer_agg_metrics.csv', index=False)
    item_metrics.to_csv(score_dir / 'transformer_item_metrics.csv', index=True)

    print("Model: transformer_model/")
    model_path = Path("transformer_model")
    model_path.mkdir(exist_ok=True)
    predictor.serialize(model_path)
    print("\n=== Saved ===")

    print(f"Metrics: {score_dir}/")
    print("="*60)