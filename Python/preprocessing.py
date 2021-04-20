'''
A library of preprocessing functions and classes for both research and ML tasks.

Matt Mauer
'''
import pandas as pd
import numpy as np
# import statsmodels.api as sm
# import seaborn as sns
# import matplotlib.pyplot as plt
import datetime as dt
# import geopandas as gpd
# import requests
# from scipy import stats

def read(path):
    'Read data from csv and prints metadata.'
    df = pd.read_csv(path)
    print(df.shape)
    df.sample(5)
    return df

def impute_missing_dates(data_series, impute_val, verbose=False):
    """Imputes missing dates in a time series with a provided value.
    In place.
    
    Inputs:
        data_series (pandas.Series): receives impute values
        impute_val any immutable data type
    """
    n_imputes = 0
    start_date = data_series.index.min()
    last_date = data_series.index.max()
    date_range = (last_date - start_date).days
    print(f"The data range from {start_date} to {last_date}.")
    for i in range(date_range+1):
        date_i = start_date + dt.timedelta(days=i)
        if date_i not in data_series.index:
            if verbose:
                print(date_i)
            data_series[date_i] = impute_val
            n_imputes += 1
            
    print(f"{n_imputes} dates have been imputed with {impute_val}.")


