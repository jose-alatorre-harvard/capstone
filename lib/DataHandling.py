

from statsmodels.tsa.stattools import adfuller
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


import pandas as pd
import numpy as np
import talib
import inspect
import datetime


def getWeights(d, lags):
    # return the weights from the series expansion of the differencing operator
    # for real orders d and up to lags coefficients
    w = [1]
    for k in range(1, lags):
        w.append(-w[-1] * ((d - k + 1)) / k)
    w = np.array(w).reshape(-1, 1)
    return w
def ts_differencing(series, order, lag_cutoff):
    # return the time series resulting from (fractional) differencing
    # for real orders order up to lag_cutoff coefficients

    weights = getWeights(order, lag_cutoff)
    res = 0
    for k in range(lag_cutoff):
        res += weights[k] * series.shift(k).fillna(0)
    return res[lag_cutoff:]

def get_fractional_stationary_series(series, lag_cutoff=100):
    interval = np.linspace(0, 1, 100)
    optimal_found = False
    for d in interval:
        ts = ts_differencing(series, order=d, lag_cutoff=lag_cutoff)
        res = adfuller(ts, maxlag=1, regression='c')  # autolag='AIC'
        adf = res[0]
        adf_limit = res[4]['5%']

        if adf <= adf_limit:
            frac_time_serie = ts
            opt_d = d
            optimal_found = True
            break

    if optimal_found == False:
        raise

    return frac_time_serie, opt_d, adf, adf_limit


def get_only_fractional_stationary_series(series, lag_cutoff=100):
    frac_time_serie, opt_d, adf, adf_limit = get_fractional_stationary_series(series, lag_cutoff=lag_cutoff)

    return frac_time_serie ,opt_d


def get_return_in_period(serie, origin_time_delta, finish_time_delta, forward_limit_time_delta,
                         return_indices=False):
    """

    :param serie:
    :param origin_time_delta:
    :param finish_time_delta:
    :return:
    """
    numerators = []
    divisors = []
    obs_dates = []
    first_date = serie.index[0] + origin_time_delta
    last_date = serie.index[-1] - forward_limit_time_delta
    for counter, i in enumerate(serie.index):
        date_numerator = i - finish_time_delta
        date_divisor = i - origin_time_delta
        if i >= first_date and i <= last_date:
            return_index_numerator = serie.index.searchsorted(date_numerator)
            return_index_divisor = serie.index.searchsorted(date_divisor)

            numerators.append(return_index_numerator)
            divisors.append(return_index_divisor)
            obs_dates.append(i)

    period_return = pd.DataFrame(index=obs_dates,
                                 data=serie.iloc[numerators].values / serie.iloc[divisors].values)
    period_return = period_return.reindex(serie.index)

    period_return = period_return.sort_index()

    try:

        index_name=serie.index.name  if serie.index.name is not None else "index"
        numerators_df=pd.DataFrame(index=obs_dates,
                                     data=serie.iloc[numerators].reset_index()[index_name].values )
    except:
        raise

    numerators_df=numerators_df.reindex(serie.index)

    numerators_df[numerators_df.columns[0]]=[i.replace(tzinfo=numerators_df.index.tzinfo) for i in numerators_df[numerators_df.columns[0]]]
    period_return = period_return.sort_index()
    if return_indices==True:
        return period_return[period_return.columns[0]] - 1 ,numerators_df , serie.iloc[divisors]
    else:
        return period_return[period_return.columns[0]] - 1


