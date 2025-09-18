import pandas as pd
import numpy as np

def add_lag_features(df, cols, lags=[1,3,6,12,24]):
    for c in cols:
        for lag in lags:
            df[f"{c}_lag{lag}"] = df[c].shift(lag)
    return df

def add_moving_averages(df, cols, windows=[3,6,12,24,168]):
    for c in cols:
        for w in windows:
            df[f"{c}_ma{w}"] = df[c].rolling(window=w, min_periods=1).mean()
    return df

def add_cyclical_time_features(df):
    hours = df.index.hour
    days = df.index.dayofweek
    months = df.index.month
    df['hour_sin'] = np.sin(2*np.pi*hours/24)
    df['hour_cos'] = np.cos(2*np.pi*hours/24)
    df['dow_sin'] = np.sin(2*np.pi*days/7)
    df['dow_cos'] = np.cos(2*np.pi*days/7)
    df['month_sin'] = np.sin(2*np.pi*(months-1)/12)
    df['month_cos'] = np.cos(2*np.pi*(months-1)/12)
    return df
