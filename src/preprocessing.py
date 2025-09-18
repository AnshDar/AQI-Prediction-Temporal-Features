import pandas as pd
import numpy as np

def load_csv(path, datetime_col='datetime'):
    df = pd.read_csv(path)
    df[datetime_col] = pd.to_datetime(df[datetime_col], infer_datetime_format=True)
    df = df.sort_values(datetime_col).reset_index(drop=True)
    df = df.set_index(datetime_col)
    return df

def interpolate_timeseries(df, freq='H', method='linear'):
    idx = pd.date_range(start=df.index.min(), end=df.index.max(), freq=freq)
    df = df.reindex(idx)
    return df.interpolate(method=method, limit_direction='both')

def train_test_split_time_series(df, test_size=0.2):
    n = len(df)
    split = int((1-test_size) * n)
    return df.iloc[:split], df.iloc[split:]
