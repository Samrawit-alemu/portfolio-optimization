import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

OUTPUT_DIR = Path("outputs/backtesting")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_assets():

    tickers = ["TSLA","SPY","BND"]

    prices = pd.DataFrame()

    for ticker in tickers:

        df = pd.read_csv(
            f"data/processed/{ticker}.csv"
        )

        df["Date"] = pd.to_datetime(df["Date"])

        df.set_index("Date", inplace=True)

        prices[ticker] = df["Close"]

    return prices

def calculate_returns(prices):

    returns = prices.pct_change()

    returns = returns.dropna()

    return returns

def portfolio_returns(returns, weights):

    weight_vector = np.array(
        list(weights.values())
    )

    portfolio = returns.dot(weight_vector)

    return portfolio

def benchmark_returns(returns):

    benchmark = (
        returns["SPY"]*0.60 +
        returns["BND"]*0.40
    )

    return benchmark
def cumulative_returns(series):

    cumulative = (1+series).cumprod()

    return cumulative

def annual_return(series):

    return (
        (1+series.mean())**252
        -1
    )

def annual_volatility(series):

    return series.std()*np.sqrt(252)

def sharpe_ratio(series):

    rf = 0.02

    excess = series-rf/252

    return (
        np.sqrt(252)
        * excess.mean()
        / excess.std()
    )

def maximum_drawdown(series):

    cumulative = (1+series).cumprod()

    peak = cumulative.cummax()

    drawdown = (
        cumulative-peak
    )/peak

    return drawdown.min()

def performance_table(strategy,benchmark):

    results = pd.DataFrame({

        "Metric":[

            "Annual Return",

            "Annual Volatility",

            "Sharpe Ratio",

            "Maximum Drawdown"

        ],

        "Strategy":[

            annual_return(strategy),

            annual_volatility(strategy),

            sharpe_ratio(strategy),

            maximum_drawdown(strategy)

        ],

        "Benchmark":[

            annual_return(benchmark),

            annual_volatility(benchmark),

            sharpe_ratio(benchmark),

            maximum_drawdown(benchmark)

        ]

    })

    return results

def performance_table(strategy,benchmark):

    results = pd.DataFrame({

        "Metric":[

            "Annual Return",

            "Annual Volatility",

            "Sharpe Ratio",

            "Maximum Drawdown"

        ],

        "Strategy":[

            annual_return(strategy),

            annual_volatility(strategy),

            sharpe_ratio(strategy),

            maximum_drawdown(strategy)

        ],

        "Benchmark":[

            annual_return(benchmark),

            annual_volatility(benchmark),

            sharpe_ratio(benchmark),

            maximum_drawdown(benchmark)

        ]

    })

    return results

def plot_backtest(strategy,benchmark):

    plt.figure(figsize=(14,6))

    plt.plot(

        cumulative_returns(strategy),

        label="Optimized Portfolio",

        linewidth=2

    )

    plt.plot(

        cumulative_returns(benchmark),

        label="Benchmark",

        linewidth=2

    )

    plt.title(

        "Portfolio Backtesting"

    )

    plt.xlabel("Date")

    plt.ylabel("Growth")

    plt.grid(True)

    plt.legend()

    plt.tight_layout()

    plt.savefig(

        OUTPUT_DIR/

        "portfolio_backtest.png",

        dpi=300

    )

    plt.show()

    def investment_recommendation(results):

    strategy = results.loc[
        results["Metric"]=="Sharpe Ratio",
        "Strategy"
    ].values[0]

    benchmark = results.loc[
        results["Metric"]=="Sharpe Ratio",
        "Benchmark"
    ].values[0]

    print("="*70)

    if strategy>benchmark:

        print("Recommendation")

        print()

        print(
            "The optimized portfolio "
            "outperformed the benchmark "
            "based on risk-adjusted return."
        )

        print()

        print(
            "Recommendation: Invest in the "
            "optimized portfolio."
        )

    else:

        print(
            "Recommendation: Benchmark "
            "performed better."
        )

    print("="*70)

    
