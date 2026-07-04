import os
import yfinance as yf
import pandas as pd


TICKERS = ["TSLA", "BND", "SPY"]
START_DATE = "2015-01-01"
END_DATE = "2026-06-30"


def download_data():
    """
    Download historical stock data and save it as CSV files.
    """

    raw_data_path = "data/raw"
    os.makedirs(raw_data_path, exist_ok=True)

    for ticker in TICKERS:
        print(f"Downloading {ticker}...")

        df = yf.download(
            ticker,
            start=START_DATE,
            end=END_DATE,
            auto_adjust=False,
            progress=False,
        )

        df.to_csv(f"{raw_data_path}/{ticker}.csv")

        print(f"{ticker} saved successfully.")

    print("\nAll datasets downloaded successfully!")


if __name__ == "__main__":
    download_data()