import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller


def adf_test(series):

    result = adfuller(series.dropna())

    print("=" * 60)
    print("ADF Statistic :", result[0])
    print("p-value :", result[1])

    if result[1] < 0.05:
        print("Result : Stationary")
    else:
        print("Result : Non-Stationary")


def difference(df):

    df = df.copy()

    df["Differenced"] = df["Close"].diff()

    return df


def plot_difference(df, ticker):

    plt.figure(figsize=(12,5))

    plt.plot(df["Date"], df["Differenced"])

    plt.title(f"{ticker} Differenced Series")

    plt.grid()

    plt.show()