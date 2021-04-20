import pandas as pd
import numpy as np
import statsmodels.api as sm
# import seaborn as sns
# import matplotlib.pyplot as plt
# import datetime as dt
# import geopandas as gpd
# import requests
# from scipy import stats


def RDD(data, threshold, running_var="X", threshold_band=2000, mini_band=250):
    '''Regression Discontinuity Design
    RDD performs a simple linear Regression Discontinuity Design test using OLS.
    Only the running variable and the indicator for treatment (which side of the
    discontinuity) are used in the model. Only the conditional effect of the 
    discontinuity is reported. Model assumes "treatment" is for values above 
    (i.e. greater than) the discontinuity. Reverse the sign of reported effects 
    to correctly interpret interventions occuring below the threshold.

    Inputs:
        data (pd.DataFrame): must have a "Y" variable that is the outcome of 
            interest.
        threshold (int or float): the value at which the discontinuity occurs.
        running_var (str): the name of the running variable in "data".
        threshold_band (int or float): the width of the band (in one direction)
            around the discontinuity for which to include the data in the RDD.
            Wider bands have higher precision but also greater bias.
        mini_band (int or float): a smaller band used to check for manipulation
            around the discontinuity.
    '''
    data['T'] = (data[running_var] >= threshold).map(int)
    n_below = data[(data[running_var] < threshold) & (data[running_var] > threshold-mini_band)].shape[0]
    n_above = data[(data[running_var] >= threshold) & (data[running_var] < threshold+mini_band)].shape[0]
    mask = (np.abs(data[running_var] - threshold) < threshold_band) 

    subset = data[mask]
    print(f'There are {subset.shape[0]} samples used in the RDD for the {threshold} threshold with a ' +
          f'band of ${threshold_band} above and below. There are {n_below} samples JUST below the cutoff and {n_above} ' + 
          f'JUST above (i.e. +/- {mini_band}).')
    X = subset[['T', running_var]]
    X = sm.add_constant(X)
    model = sm.OLS(subset.Y, X)
    results = model.fit()
    print(results.t_test('T = 0'))

