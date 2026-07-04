import numpy as np


def calculate_returns(df):

    df["Daily Return"]=df["Close"].pct_change()

    return df


def sharpe_ratio(df,risk_free_rate=0.02):

    returns=df["Daily Return"].dropna()

    excess=returns-risk_free_rate/252

    return np.sqrt(252)*excess.mean()/excess.std()


def value_at_risk(df,confidence=0.95):

    returns=df["Daily Return"].dropna()

    return np.percentile(returns,(1-confidence)*100)