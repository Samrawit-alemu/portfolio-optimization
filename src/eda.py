import matplotlib.pyplot as plt
import pandas as pd


def plot_close_price(df, ticker):

    plt.figure(figsize=(12,5))

    plt.plot(df["Date"], df["Close"])

    plt.title(f"{ticker} Closing Price")

    plt.xlabel("Date")

    plt.ylabel("Price")

    plt.grid(True)

    plt.show()


def plot_volume(df, ticker):

    plt.figure(figsize=(12,5))

    plt.plot(df["Date"], df["Volume"])

    plt.title(f"{ticker} Trading Volume")

    plt.grid(True)

    plt.show()


def daily_return(df):

    df = df.copy()

    df["Daily Return"] = df["Close"].pct_change()

    return df


def plot_daily_return(df, ticker):

    plt.figure(figsize=(12,5))

    plt.plot(df["Date"], df["Daily Return"])

    plt.title(f"{ticker} Daily Return")

    plt.grid(True)

    plt.show()


def rolling_statistics(df):

    df = df.copy()

    df["Rolling Mean"] = df["Close"].rolling(30).mean()

    df["Rolling Std"] = df["Close"].rolling(30).std()

    return df


def plot_rolling(df, ticker):

    plt.figure(figsize=(12,5))

    plt.plot(df["Date"], df["Close"], label="Close")

    plt.plot(df["Date"], df["Rolling Mean"], label="Rolling Mean")

    plt.legend()

    plt.grid(True)

    plt.title(ticker)

    plt.show()