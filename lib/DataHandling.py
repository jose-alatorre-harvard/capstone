

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

    numerators_df=pd.DataFrame(index=obs_dates,
                                 data=serie.iloc[numerators].reset_index()["index"].values )

    numerators_df=numerators_df.reindex(serie.index)

    period_return = period_return.sort_index()
    if return_indices==True:
        return period_return[period_return.columns[0]] - 1 ,numerators_df , serie.iloc[divisors]
    else:
        return period_return[period_return.columns[0]] - 1


class DailyDataFrame2Features:
    """

    """

    def __init__(self, bars_dict, configuration_dict, features_list=None, exclude_features=None,forward_returns_time_delta=None ):
        """
        feature_list and exclude_list should be by column
        :param bars_dict: keys:asset_name, value:pd.DataFrame bars
        :param features_list:
        :param exclude_features:
        :param configuration_dict
        """

        #all time series should have the same time index
        for counter,ts in enumerate(bars_dict.values()):
            if counter ==0:
                base_index=ts.index
            else:
                assert base_index.equals(ts.index)

        self.configuration_dict = configuration_dict
        all_features = pd.DataFrame()
        for asset_name, bars_time_serie_df in bars_dict.items():
            features_instance = DailySeries2Features(bars_time_serie_df, features_list, exclude_features,
                                                     forward_returns_time_delta)
            technical_features = features_instance.technical_features.copy()
            technical_features.columns = [asset_name + "_" + i for i in technical_features.columns]
            all_features = pd.concat([all_features, technical_features], axis=1)

        #set forward_returns
        if forward_returns_time_delta is not None:
            self.forward_returns_dates=features_instance.forward_returns_dates


        # we drop N/A because there are no features available on certains dates like moving averages
        self.all_features = all_features#.dropna()

        self.windsorized_data = self.all_features.clip(lower=self.all_features.quantile(q=.025),
                                                       upper=self.all_features.quantile(q=.975),
                                                       axis=1)

    def separate_features_from_forward_returns(self,features):
        """
        separates input features from forward returns
        :param features:
        :return:
        """
        only_features=features[[col for col in features.columns if "forward_return" not in col]]
        only_forwad_returns=features[[col for col in features.columns if "forward_return"  in col]]

        return only_features, only_forwad_returns

    def create_pca_projection(self, exclude_feature_columns, var_limit=.02):
        """
        Create PCA features
        :param exclude_feature_columns:
        :return:
        """
        # todo: exclude features that shouldnt be take in count for PCA like categorical

        # windsorize data as PCA is sensitive

        # scale and transform
        std_clf = make_pipeline(StandardScaler(), PCA(n_components="mle"))
        pca_projection = std_clf.fit_transform(self.windsorized_data)
        explained_variance = std_clf["pca"].explained_variance_ratio_
        # Just keep features that explain more than 2% of the data
        pca_projection = pca_projection[[counter for counter, i in enumerate(explained_variance) if i > var_limit]]
        return pca_projection


class DailySeries2Features:
    """
    Adds features that require closing date data
    """

    RSI_TIME_FRAME = 14
    BOLLINGER_TIME_FRAME = 21
    EWMA_VOL_ALPHA = .98
    ANUALIZING_FACTOR = 252

    def __init__(self, serie_or_df, features_list=None, exclude_features=None, forward_returns_time_delta=None):
        """

        :param serie: pandas.Serie
        :param serie_or_df:
        :param features_list:
        :param exclude_features:
        :param forward_returns_time_delta:
        """

        if exclude_features == None:
            exclude_features = []

        self.feature_list = features_list

        if isinstance(serie_or_df, pd.Series):
            serie = serie_or_df
        else:
            serie = serie_or_df["close"]

        self.technical_features = pd.DataFrame(index=serie.index)
        self.log_prices = np.log(serie)
        self.forward_returns_dates=[]

        if features_list is not None:

            self._set_features(features_list=features_list)
            raise NotImplementedError
        else:
            for method in inspect.getmembers(self, predicate=inspect.ismethod):

                feature_name = method[0].replace("_add_", "")

                if not "_add_" + feature_name in exclude_features:

                    if "_add_" in method[0]:

                        technical = method[1](serie)
                        self._update_feature(technical=technical, feature_name=feature_name)
                    elif "_addhlc_" in method[0]:
                        # methods that require high low and close
                        if isinstance(serie_or_df, pd.DataFrame):
                            technical = method[1](serie_or_df)
                            self._update_feature(technical=technical, feature_name=feature_name)

        if forward_returns_time_delta is not None:
            # add forward returns
            for forward_td in forward_returns_time_delta:
                feature = self._set_forward_return(serie=serie, forward_return_td=forward_td)
                feature_name = feature.name
                self._update_feature(technical=feature, feature_name=feature_name)

    def _set_forward_return(self, serie, forward_return_td):
        """
        adds a forward return
        :param forward_return_td:
        :return:
        """
        origin_time_delta = datetime.timedelta(days=0)
        finish_time_delta = -forward_return_td
        forward_limit_time_delta = forward_return_td
        forward_return,numerators_df,denominators = get_return_in_period(serie, origin_time_delta, finish_time_delta, forward_limit_time_delta,
                                              return_indices=True)

        fwd_r_name=("forward_return_" + str(forward_return_td)).replace(" ", "_")

        numerators_df.columns=[fwd_r_name]

        self.forward_returns_dates.append(numerators_df)
        forward_return.name = fwd_r_name

        return forward_return

    def _update_feature(self, technical, feature_name):
        try:
            if isinstance(technical, pd.DataFrame):
                self.technical_features = pd.concat([self.technical_features, technical], axis=1)
            else:
                self.technical_features[feature_name] = technical
        except:
            raise

    def _set_features(self, features_list):

        for feature in features_list:

            try:
                getattr(self, "_add_" + feature)
            except:
                print("feature " + feature + "  not found")

    def _add_rsi(self, serie):
        technical = talib.RSI(serie, self.RSI_TIME_FRAME)
        return technical

    def _add_bollinger_bands(self, serie):
        technical = talib.BBANDS(serie, self.BOLLINGER_TIME_FRAME)
        technical = pd.DataFrame(technical).T
        technical.columns = ["bollinger_up", "bollinger_mid", "bollinger_low"]

        return technical

    def _add_ewma_vol(self, serie):

        techinical = serie.ewm(alpha=self.EWMA_VOL_ALPHA).std() * np.sqrt(self.ANUALIZING_FACTOR)
        return techinical

    def _add_50_days_ma(self, serie):
        """
        moving average is normalized to last close value to be comparable
        :param serie:
        :return:
        """
        techinical = (serie.rolling(50).mean()).divide(serie)
        return techinical

    def _add_100_days_ma(self, serie):
        """
       moving average is normalized to last close value to be comparable
       :param serie:
       :return:
       """
        techinical = (serie.rolling(100).mean()).divide(serie)
        return techinical

    def _add_200_days_ma(self, serie):
        """
       moving average is normalized to last close value to be comparable
       :param serie:
       :return:
       """
        techinical = (serie.rolling(200).mean()).divide(serie)
        return techinical

    def _add_fraction_diff(self, serie):
        # Todo: Return optimal d rolling to make it full not future dependant
        frac_time_serie, opt_d, adf, adf_limit = get_fractional_stationary_series(serie, lag_cutoff=100)
        self.fractional_diff_optimal_d = opt_d
        return frac_time_serie

    def _add_log_returns(self, serie):
        feature = self.log_prices.copy().diff()
        return feature

    # def _addhlc_natr(self,data_frame):
    #     pass

    def _add_12m1_past_return(self, serie):
        """
        return on the last 12 months ignoring last month
        :param data_frame:
        :return:
        """
        origin_time_delta = datetime.timedelta(days=365)
        finish_time_delta = datetime.timedelta(days=30)
        forward_limit_time_delta = datetime.timedelta(days=0)
        technical = get_return_in_period(serie, origin_time_delta, finish_time_delta, forward_limit_time_delta)

        return technical

    def _add_3m_past_return(self, serie):
        """
        return on the last 12 months ignoring last month
        :param data_frame:
        :return:
        """
        origin_time_delta = datetime.timedelta(days=365)
        finish_time_delta = datetime.timedelta(days=30)
        forward_limit_time_delta = datetime.timedelta(days=0)
        technical = get_return_in_period(serie, origin_time_delta, finish_time_delta, forward_limit_time_delta)

        return technical

    def _add_1m_past_return(self, serie):
        """
        return on the last 12 months ignoring last month
        :param data_frame:
        :return:
        """
        origin_time_delta = datetime.timedelta(days=30)
        finish_time_delta = datetime.timedelta(days=0)
        forward_limit_time_delta = datetime.timedelta(days=0)
        technical = get_return_in_period(serie, origin_time_delta, finish_time_delta, forward_limit_time_delta)

        return technical

    def _add_3m_past_return(self, serie):
        """
        return on the last 12 months ignoring last month
        :param data_frame:
        :return:
        """
        origin_time_delta = datetime.timedelta(days=90)
        finish_time_delta = datetime.timedelta(days=0)
        forward_limit_time_delta = datetime.timedelta(days=0)
        technical = get_return_in_period(serie, origin_time_delta, finish_time_delta, forward_limit_time_delta)

        return technical

