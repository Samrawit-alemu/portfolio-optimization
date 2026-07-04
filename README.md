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
