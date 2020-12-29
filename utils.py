import datetime
import inspect
import numpy as np
import os
import pandas as pd
import talib
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from timeseriescv.cross_validation import CombPurgedKFoldCV


def get_return_in_period(serie, origin_time_delta, finish_time_delta, forward_limit_time_delta,
                         return_indices=False):
    """
    calculates the return in the specified time period
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



class DailyDataFrame2Features:
    """
    Convert daily data into features
    """

    def __init__(self, bars_dict, configuration_dict, features_list=None, exclude_features=None,forward_returns_time_delta=None ):
        """
        feature_list and exclude_list should be by column
        :param bars_dict: keys:asset_name, value:pd.DataFrame bars
        :param features_list:
        :param exclude_features:
        :param configuration_dict
        """

        # check that all time series have the same time index
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

        # we drop N/A because there are no features available on certains dates like moving averages
        self.all_features = all_features.dropna()


        # set forward returns
        if forward_returns_time_delta is not None:
            self.forward_returns_dates = features_instance.forward_returns_dates[0]
            self.forward_returns_dates = self.forward_returns_dates.reindex(self.all_features.index)

        self.windsorized_data = self.all_features.clip(lower=self.all_features.quantile(q=.025),
                                                       upper=self.all_features.quantile(q=.975),
                                                       axis=1)

    def add_lags_to_features(self,features,n_lags):
        """
        Adds lags as columns for each column in features
        :param feautures: pandas.DataFrame,
        :param n_lagas: int
        :return:
        """
        original_features=features.copy()
        new_features=features.copy()
        assert n_lags >=1
        for lag in range(n_lags):
            shifted_df=original_features.shift(lag+1)
            shifted_df.columns=[i+"_lag_"+str(lag+1) for i in shifted_df.columns]
            new_features=pd.concat([new_features,shifted_df],axis=1)

        return new_features

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
        create PCA features
        :param exclude_feature_columns:
        :return:
        """
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
        """
        calculates the relative strength index
        :param serie:
        :return:
        """
        technical = talib.RSI(serie, self.RSI_TIME_FRAME)
        return technical

    def _add_bollinger_bands(self, serie):
        """
        calculates the Bollinger Bands
        :param serie:
        :return:
        """
        technical = talib.BBANDS(serie, self.BOLLINGER_TIME_FRAME)
        technical = pd.DataFrame(technical).T
        technical.columns = ["bollinger_up", "bollinger_mid", "bollinger_low"]

        return technical

    def _add_ewma_vol(self, serie):
        """
        calculates the exponentially weighted moving average (EWMA)
        :param serie:
        :return:
        """
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


    def _add_log_returns(self, serie):
        """
        taking the log of the returns
        :param serie:
        :return:
        """

        feature = self.log_prices.copy().diff()

        return feature

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

    def _add_detrend(self, serie):
        """
        detrend the asset prices and scale
        :param data_frame:
        :return:
        """
        holtwinters = ExponentialSmoothing(serie, trend="add", seasonal=None)
        fit = holtwinters.fit(optimized=True, remove_bias=True, smoothing_level=0.1, smoothing_slope=0.1)

        hw_trend = np.array(fit._results.trend).reshape(-1, 1)
        hw_level = np.array(fit._results.level).reshape(-1, 1)
        hw_resid = np.array(fit._results.resid).reshape(-1, 1)
        print("hw_trend pre scaled", hw_trend)
        print("hw_level pre scaled", hw_level)
        print("hw_resid pre scaled", hw_resid)

        trend_scaler = preprocessing.MinMaxScaler().fit(hw_trend)
        trend_scaled = pd.Series(trend_scaler.transform(hw_trend).flatten())
        print("hw_trend post scaled", trend_scaled)

        level_scaler = preprocessing.MinMaxScaler().fit(hw_level)
        level_scaled = pd.Series(level_scaler.transform(hw_level).flatten())
        print("hw_level post scaled", level_scaled)

        resid_scaler = preprocessing.MinMaxScaler().fit(hw_resid)
        resid_scaled = pd.Series(resid_scaler.transform(hw_resid).flatten())
        print("hw_resid post scaled", resid_scaled)

        demeaned_resid = fit._results.resid - np.mean(fit._results.resid)
        demeaned_return = np.diff(demeaned_resid, prepend=demeaned_resid[0]).reshape(-1, 1)
        print("demeaned return scaled", demeaned_return)

        demeaned_return_scaler = preprocessing.MinMaxScaler().fit(demeaned_return)
        demeaned_return_scaled = pd.Series(demeaned_return_scaler.transform(demeaned_return).flatten())
        print("demeaned return post scaled", demeaned_return_scaled)

        volatility_resid =pd.Series(fit._results.resid).ewm(alpha=self.EWMA_VOL_ALPHA, min_periods=14).var().values.reshape(-1, 1)
        volatility_resid_scaler = preprocessing.MinMaxScaler().fit(volatility_resid)
        volatility_resid_scaled = pd.Series(volatility_resid_scaler.transform(volatility_resid).flatten())
        print("volatility pre scaled", volatility_resid)
        print("volatility post scaled", volatility_resid_scaled)


        technical = pd.concat([trend_scaled, level_scaled, resid_scaled, demeaned_return_scaled, volatility_resid_scaled], axis=1)

        technical.columns = ["hw_trend", "hw_level", "hw_resid", "hw_demeaned_return", "hw_volatility"]
        technical.index = serie.index
        return technical


def build_and_persist_features_from_dir( meta_parameters, data_hash,
                                              data_dir="data_env",):
    """
    Do transformations that shouldn't be part of the class
    Also uses the meta parameters
    """

    assets_dict = {file: pd.read_parquet(data_dir + "/" + file).first() for file in
                   os.listdir(data_dir)}
    counter = 0
    for key, value in assets_dict.items():
        if counter == 0:
            main_index = value.index
        else:
            main_index = main_index.join(value.index, how="inner")

    for key, value in assets_dict.items():
        tmp_df = value.reindex(main_index)
        tmp_df = tmp_df.fillna(method='ffill')
        assets_dict[key] = tmp_df

    build_and_persist_features(assets_dict=assets_dict, in_bars_count=meta_parameters["in_bars_count"],
                               out_reward_window=meta_parameters["out_reward_window"],data_hash=data_hash)

def build_and_persist_features(assets_dict, out_reward_window,in_bars_count,data_hash):
    """
    builds close-to-close returns for a specified dataset
    :param assets_dict:
    :param out_reward_window:
    :param in_bars_count:
    :param data_hash:
    :return:
    """

    PERSISTED_DATA_DIRECTORY = "temp_persisted_data"
    if not os.path.exists(PERSISTED_DATA_DIRECTORY + "/only_features_"+data_hash):

        features_instance=DailyDataFrame2Features(bars_dict=assets_dict
                                                  ,configuration_dict={},
                                                  forward_returns_time_delta=[out_reward_window])
        features=features_instance.all_features
        only_features, only_forward_returns =features_instance.separate_features_from_forward_returns(features=features)
        forward_returns_dates = features_instance.forward_returns_dates
        only_features=only_features[[col for col in only_features.columns if "log_return" in col]]

        # get the lagged returns as features
        only_features=features_instance.add_lags_to_features(only_features,n_lags=in_bars_count)
        only_features=only_features.dropna()
        only_forward_returns=only_forward_returns.reindex(only_features.index)
        forward_returns_dates=forward_returns_dates.reindex(only_features.index)

        # Add bias to features
        only_features["bias"] = 1
        only_features.to_parquet(PERSISTED_DATA_DIRECTORY + "/only_features_" + data_hash)
        only_forward_returns.to_parquet(PERSISTED_DATA_DIRECTORY + "/only_forward_returns_" + data_hash)
        forward_returns_dates.to_parquet(PERSISTED_DATA_DIRECTORY + "/forward_return_dates_" + data_hash)

    else:
        print("features already persisted")

def train_val_test_purge_combinatorial_kfold(data_as_supervised_df,y,eval_times_df,n_splits,n_test_splits,embargo_td,test_train_percent_split):
    """
    performs a train/val/test split using combinatorial purged cross-validation
    :param data_as_supervised_df:(pandas.DataFrame) DataFrame with data as supervised index=observation_date,columns=features/state
    :param y: (pandas.Serie) predictions or rewards , index=observation_date
    :param eval_times_df: (pandas.DataFrame)  time when the reward is obtained (index=prediction_time,evaluation_time)
    :param n_splits:
    :param embargo:
    :param test_train_percent_split:
    :return:
    """
    last_index=int(data_as_supervised_df.shape[0]*test_train_percent_split)
    X_train=data_as_supervised_df.iloc[:last_index]
    X_test=data_as_supervised_df.iloc[last_index:]
    y_train = y.iloc[:last_index]
    y_test = y.iloc[last_index:]
    eval_times_df_train = eval_times_df.iloc[:last_index]
    eval_times_df_test = eval_times_df.iloc[last_index:]

    #time when the prediction is done corresponds to indes of supervised
    pred_times = pd.Series(index=X_train.index, data=X_train.index)

    tmp_combinatorial_purge = CombPurgedKFoldCV(n_splits=n_splits, n_test_splits=n_test_splits,
                                                embargo_td=embargo_td)

    splits_generator=tmp_combinatorial_purge.split(X=X_train,y=y_train,pred_times=pred_times,eval_times=eval_times_df_train)

    return splits_generator, X_test,y_test

def train_test(portfolio_df, y, return_dates):
    """
    Performs a train/test split
    :param feature_df:
    :param returns_df:
    :param return_dates_df:
    :param weights:
    :return:
    """

    features = portfolio_df
    features.index = pd.to_datetime(features.index)

    eval_times_df = pd.Series(return_dates.values, index=portfolio_df.index)
    eval_times_df = pd.to_datetime(eval_times_df)
    eval_times_df.index = pd.to_datetime(eval_times_df.index)

    data_as_supervised_df = features
    test_train_percent_split = 0.7

    embargo_td = 1

    last_index=int(data_as_supervised_df.shape[0]*test_train_percent_split)
    X_train=data_as_supervised_df.iloc[:last_index-embargo_td]
    X_test=data_as_supervised_df.iloc[last_index:]

    X_train.to_csv('temp_persisted_data/X_train.csv')
    X_test.to_csv('temp_persisted_data/X_test.csv')

    return last_index, embargo_td