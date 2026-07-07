import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from pypfopt import expected_returns
from pypfopt import risk_models
from pypfopt import EfficientFrontier
from pypfopt import plotting

OUTPUT_DIR = Path("outputs/portfolio")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =====================================================
# LOAD DATA
# =====================================================

def load_assets():

    tickers = ["TSLA", "SPY", "BND"]

    prices = pd.DataFrame()

    for ticker in tickers:

        df = pd.read_csv(
            f"data/processed/{ticker}.csv"
        )

        df["Date"] = pd.to_datetime(df["Date"])

        df.set_index("Date", inplace=True)

        prices[ticker] = df["Close"]

    return prices


# =====================================================
# EXPECTED RETURNS
# =====================================================

def calculate_expected_returns(prices):

    mu = expected_returns.mean_historical_return(
        prices
    )

    return mu


# =====================================================
# COVARIANCE MATRIX
# =====================================================

def covariance_matrix(prices):

    cov = risk_models.sample_cov(prices)

    return cov

# =====================================================
# MAXIMUM SHARPE
# =====================================================

def maximum_sharpe(mu, cov):

    ef = EfficientFrontier(mu, cov)

    weights = ef.max_sharpe()

    cleaned = ef.clean_weights()

    performance = ef.portfolio_performance(
        verbose=True
    )

    return ef, cleaned, performance


# =====================================================
# MINIMUM VOLATILITY
# =====================================================

def minimum_volatility(mu, cov):

    ef = EfficientFrontier(mu, cov)

    weights = ef.min_volatility()

    cleaned = ef.clean_weights()

    performance = ef.portfolio_performance(
        verbose=True
    )

    return ef, cleaned, performance

# =====================================================
# PLOT EFFICIENT FRONTIER
# =====================================================

def plot_frontier(mu, cov):

    ef = EfficientFrontier(mu, cov)

    fig, ax = plt.subplots(figsize=(10,7))

    plotting.plot_efficient_frontier(
        ef,
        ax=ax,
        show_assets=True
    )

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR /
        "efficient_frontier.png",
        dpi=300
    )

    plt.show()

    # =====================================================
# COVARIANCE HEATMAP
# =====================================================

def covariance_heatmap(cov):

    plt.figure(figsize=(7,6))

    plt.imshow(
        cov,
        interpolation="nearest"
    )

    plt.colorbar()

    plt.xticks(
        range(len(cov.columns)),
        cov.columns
    )

    plt.yticks(
        range(len(cov.columns)),
        cov.columns
    )

    plt.title(
        "Covariance Matrix"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR /
        "covariance_matrix.png",
        dpi=300
    )

    plt.show()

    # =====================================================
# PRINT WEIGHTS
# =====================================================

def print_weights(title, weights):

    print()

    print("="*60)

    print(title)

    print("="*60)

    for asset, weight in weights.items():

        print(
            f"{asset:5s} : {weight:.2%}"
        )

    print("="*60)

    