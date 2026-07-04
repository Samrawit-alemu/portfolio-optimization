import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller

from pmdarima import auto_arima

from statsmodels.tsa.arima.model import ARIMA

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

import numpy as np


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

def split_data(df):

    train_size=int(len(df)*0.8)

    train=df["Close"][:train_size]

    test=df["Close"][train_size:]

    return train,test
def train_model(train):

    auto_model=auto_arima(
        train,
        seasonal=False,
        trace=True,
        suppress_warnings=True
    )

    order=auto_model.order

    print("Best Order:",order)

    model=ARIMA(train,order=order)

    fitted=model.fit()

    return fitted

def evaluate(model,test):

    forecast=model.forecast(len(test))

    mae=mean_absolute_error(test,forecast)

    rmse=np.sqrt(mean_squared_error(test,forecast))

    print("MAE :",mae)

    print("RMSE :",rmse)

    return forecast

def plot_forecast(train,test,forecast,ticker):

    plt.figure(figsize=(12,5))

    plt.plot(train.index,train,label="Train")

    plt.plot(test.index,test,label="Actual")

    plt.plot(test.index,forecast,label="Forecast")

    plt.title(ticker)

    plt.legend()

    plt.grid()

    plt.show()
