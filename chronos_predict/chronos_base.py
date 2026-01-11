import pandas as pd
from ch_function import *

ALL_PREDICT = False

def main():
    print("=== Loading Data ===")
    df = pd.read_parquet('data/df_for.parquet')
    print("Complete!")

    decile_books = extract_decile_books(df)

    pipeline, accelerator = load_model()
    samples = preprocess_data(df)
    print(f"Prepared {len(samples)} valid samples.")
    forecasts = run_inference(pipeline, samples, accelerator)
    save_results(samples, forecasts, decile_books, ALL_PREDICT)
    print("Inference and saving complete.")

if __name__ == "__main__":
    main()
