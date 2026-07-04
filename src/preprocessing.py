import os
import pandas as pd


RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"


def load_data(ticker):
    path = os.path.join(RAW_DATA_DIR, f"{ticker}.csv")
    return pd.read_csv(path)


def inspect_data(df, ticker):
    print("=" * 60)
    print(f"{ticker} Dataset")
    print("=" * 60)

    print("\nFirst 5 rows")
    print(df.head())

    print("\nShape")
    print(df.shape)

    print("\nData Types")
    print(df.dtypes)

    print("\nMissing Values")
    print(df.isnull().sum())

    print("\nSummary Statistics")
    print(df.describe())


def clean_data(df):

    df = df.copy()

    df.drop_duplicates(inplace=True)

    df["Date"] = pd.to_datetime(df["Date"])

    df.sort_values("Date", inplace=True)

    df.reset_index(drop=True, inplace=True)

    df.ffill(inplace=True)

    return df


def save_processed(df, ticker):

    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    path = os.path.join(PROCESSED_DATA_DIR, f"{ticker}.csv")

    df.to_csv(path, index=False)

    print(f"{ticker} processed data saved.")