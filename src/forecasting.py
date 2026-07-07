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

# ==========================================================
# FUTURE FORECAST
# ==========================================================

def forecast_future(model, periods=252):
    """
    Forecast future prices.

    252 trading days ≈ 12 months
    126 trading days ≈ 6 months
    """

    forecast = model.get_forecast(steps=periods)

    future = forecast.predicted_mean

    confidence = forecast.conf_int()

    return future, confidence


# ==========================================================
# PLOTTING FUNCTIONS
# ==========================================================

def plot_test_prediction(train, test, prediction, ticker):

    plt.figure(figsize=(15,6))

    plt.plot(train.index, train.values,
             label="Train", linewidth=2)

    plt.plot(test.index, test.values,
             label="Actual", linewidth=2)

    plt.plot(test.index, prediction,
             label="Prediction", linewidth=2)

    plt.title(f"{ticker} Test Forecast")

    plt.xlabel("Date")

    plt.ylabel("Price")

    plt.grid(True)

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        FORECAST_DIR / f"{ticker}_test_prediction.png",
        dpi=300
    )

    plt.show()


def plot_future_forecast(
        df,
        future,
        confidence,
        ticker):

    plt.figure(figsize=(15,6))

    plt.plot(
        df.index,
        df["Close"],
        label="Historical",
        linewidth=2
    )

    future_dates = pd.date_range(
        start=df.index[-1] + pd.Timedelta(days=1),
        periods=len(future),
        freq="B"
    )

    plt.plot(
        future_dates,
        future,
        label="Forecast",
        linewidth=2,
        color="red"
    )

    plt.fill_between(
        future_dates,
        confidence.iloc[:,0],
        confidence.iloc[:,1],
        alpha=0.30,
        label="95% Confidence Interval"
    )

    plt.title(f"{ticker} Future Forecast")

    plt.xlabel("Date")

    plt.ylabel("Price")

    plt.grid(True)

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        FORECAST_DIR / f"{ticker}_future_forecast.png",
        dpi=300
    )

    plt.show()

    # ==========================================================
# FORECAST SUMMARY
# ==========================================================

def summarize_forecast(future):

    first_price = future.iloc[0]

    last_price = future.iloc[-1]

    expected_return = (
        (last_price - first_price)
        / first_price
    )

    print("="*60)

    print("Forecast Summary")

    print("="*60)

    print(f"Forecast Start Price : {first_price:.2f}")

    print(f"Forecast End Price   : {last_price:.2f}")

    print(f"Expected Return      : {expected_return:.2%}")

    print("="*60)

    return expected_return

# ==========================================================
# TREND ANALYSIS
# ==========================================================

def analyze_trend(future):

    start = future.iloc[0]

    end = future.iloc[-1]

    if end > start:

        trend = "Upward Trend"

    elif end < start:

        trend = "Downward Trend"

    else:

        trend = "Stable Trend"

    return trend


def confidence_analysis(confidence):

    width = (
        confidence.iloc[:,1]
        - confidence.iloc[:,0]
    )

    if width.iloc[-1] > width.iloc[0]:

        message = (
            "Confidence interval widens over time, "
            "indicating increasing uncertainty "
            "for long-term forecasts."
        )

    else:

        message = (
            "Confidence interval remains relatively "
            "stable over the forecast horizon."
        )

    return message


def market_insight(expected_return):

    if expected_return > 0.15:

        opportunity = (
            "Strong expected growth. Potential buying opportunity."
        )

    elif expected_return > 0:

        opportunity = (
            "Moderate expected growth."
        )

    else:

        opportunity = (
            "Negative expected return."
        )

    return opportunity


def market_risk(confidence):

    width = (
        confidence.iloc[:,1]
        - confidence.iloc[:,0]
    ).mean()

    if width > 50:

        risk = (
            "High uncertainty and volatility."
        )

    elif width > 20:

        risk = (
            "Moderate uncertainty."
        )

    else:

        risk = (
            "Relatively stable forecast."
        )

    return risk