# Portfolio Optimization

## Overview

This project analyzes historical financial data from Tesla (TSLA), Vanguard Total Bond ETF (BND), and SPY to perform financial analysis and build a baseline forecasting model.

## Project Structure

```
data/
    raw/
    processed/

notebooks/

src/

tests/
```

## Features

- Historical data extraction using Yahoo Finance
- Data preprocessing
- Exploratory Data Analysis
- Rolling statistics
- Stationarity testing using Augmented Dickey-Fuller Test
- Risk metrics
  - Daily Returns
  - Sharpe Ratio
  - Value at Risk
- ARIMA forecasting model

## Technologies

- Python
- Pandas
- NumPy
- Matplotlib
- Statsmodels
- pmdarima
- scikit-learn

## Dataset

- TSLA
- SPY
- BND

Historical Period:

2015-01-01 → 2026-06-30

## Results

The project successfully:

- Collected financial data from Yahoo Finance
- Performed preprocessing and exploratory analysis
- Conducted stationarity testing
- Built an ARIMA forecasting model
- Forecasted future prices with confidence intervals
- Optimized the investment portfolio using Modern Portfolio Theory
- Evaluated the optimized portfolio against a benchmark through backtesting

## Future Improvements

- Compare ARIMA with LSTM and Prophet
- Include additional ETFs and stocks
- Perform rolling-window backtesting
- Build an interactive dashboard
