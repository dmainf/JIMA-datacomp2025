import pandas as pd
from pytrends.request import TrendReq
import time
from datetime import datetime

def extract_google_trends(test_mode=False, num_titles=5):
    df = pd.read_csv('normalized_title(remove).csv')

    titles = df['normalized_title'].apply(lambda x: x.split('_')[0]).unique()

    if test_mode:
        titles = titles[:num_titles]
        print(f"Test mode: Processing only {len(titles)} titles")

    pytrends = TrendReq(hl='ja-JP', tz=360)

    timeframe = '2023-01-01 2024-12-31'

    all_data = {}
    date_range = None

    for i, title in enumerate(titles):
        try:
            print(f"Processing {i+1}/{len(titles)}: {title}")

            pytrends.build_payload([title], timeframe=timeframe, geo='JP')
            interest_over_time_df = pytrends.interest_over_time()

            if not interest_over_time_df.empty:
                interest_over_time_df = interest_over_time_df.drop(columns=['isPartial'], errors='ignore')
                all_data[title] = interest_over_time_df[title]
                if date_range is None:
                    date_range = interest_over_time_df.index
                print(f"  ✓ Data collected for {title}")
            else:
                print(f"  ✗ No data available for {title}")
                all_data[title] = None

            time.sleep(2)

        except Exception as e:
            print(f"Error processing {title}: {e}")
            all_data[title] = None
            time.sleep(5)
            continue

    if date_range is not None:
        final_df = pd.DataFrame(index=date_range)
        for title in titles:
            if all_data.get(title) is not None:
                final_df[title] = all_data[title]
            else:
                final_df[title] = 0

        final_df.to_csv('google_trends_data.csv')
        print(f"Saved data for {len(titles)} titles to google_trends_data.csv")
    else:
        print("No data collected")

if __name__ == '__main__':
    extract_google_trends()
