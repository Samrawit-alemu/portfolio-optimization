import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
)

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA

from pmdarima import auto_arima


# ==========================================================
# PATHS
# ==========================================================

PROCESSED_DATA_DIR = Path("data/processed")
FORECAST_DIR = Path("outputs/forecast")

FORECAST_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================================
# DATA LOADING
# ==========================================================

def load_dataset(ticker: str):

    file = PROCESSED_DATA_DIR / f"{ticker}.csv"

    df = pd.read_csv(file)

    df["Date"] = pd.to_datetime(df["Date"])

    df.sort_values("Date", inplace=True)

    df.set_index("Date", inplace=True)

    return df


# ==========================================================
# STATIONARITY
# ==========================================================

def adf_test(series):

    result = adfuller(series.dropna())

    print("=" * 60)

    print("ADF Statistic :", result[0])

    print("P-value :", result[1])

    print()

    if result[1] < 0.05:
        print("Series is Stationary")
    else:
        print("Series is NOT Stationary")

    print("=" * 60)

    return result


def difference(series):

    return series.diff().dropna()


# ==========================================================
# TRAIN TEST SPLIT
# ==========================================================

def split_data(df, train_ratio=0.80):

    n = int(len(df) * train_ratio)

    train = df.iloc[:n]

    test = df.iloc[n:]

    return train, test


# ==========================================================
# AUTO ARIMA
# ==========================================================

def find_best_order(train):

    print("Searching best ARIMA order...")

    model = auto_arima(
        train,
        seasonal=False,
        trace=True,
        suppress_warnings=True,
        error_action="ignore",
        stepwise=True,
    )

    print()

    print("Best Order :", model.order)

    return model.order


# ==========================================================
# MODEL TRAINING
# ==========================================================

def train_arima(train):

    order = find_best_order(train)

    model = ARIMA(train, order=order)

    fitted = model.fit()

    return fitted


# ==========================================================
# TEST FORECAST
# ==========================================================

def predict_test(model, test):

    forecast = model.get_forecast(steps=len(test))

    prediction = forecast.predicted_mean

    confidence = forecast.conf_int()

    return prediction, confidence


# ==========================================================
# EVALUATION
# ==========================================================

def evaluate(test, prediction):

    mae = mean_absolute_error(test, prediction)

    rmse = np.sqrt(mean_squared_error(test, prediction))

    mape = mean_absolute_percentage_error(test, prediction)

    print("=" * 60)

    print("MAE :", round(mae, 4))

    print("RMSE :", round(rmse, 4))

    print("MAPE :", round(mape, 4))

    print("=" * 60)

    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
    }