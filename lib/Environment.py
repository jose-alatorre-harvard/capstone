import gym
import pandas as pd
import numpy as np
import os
import datetime
from tqdm import tqdm
from lib.Benchmarks import SimulatedAsset
from lib.DataHandling import DailyDataFrame2Features

class State:

    def __init__(self, features,forward_returns,forward_returns_dates, in_bars_count, objective_parameters):
        """

        :param features: (dict)
        :param in_bars_count: (int)
        :param objective_parameters:(dict)

        """

        self.features = features
        self.forward_returns=forward_returns
        self.forward_returns_dates=forward_returns_dates
        self.in_bars_count = in_bars_count
        self._set_helper_functions()
        self._set_objective_function_parameters(objective_parameters)


        self._initialize_weights_buffer()


    def _set_helper_functions(self):
        """
        Creates following properties
        assets_names: (list)
        log_close_returns: (pd.DataFrame)
        :return:
        """

        self.number_of_assets=len(self.forward_returns.columns)
        self.state_features_shape=(self.in_bars_count,self.features.shape[1])

    def _set_objective_function_parameters(self,objective_parameters):
        self.percent_commission = objective_parameters["percent_commission"]

    def reset(self):
        """
        resets the weights_buffer

        """

        self._initialize_weights_buffer()


    @property
    def asset_names(self):
        """
               Todo: Make proper parsing
               :return:
               """
        return self.forward_returns.columns

    def _initialize_weights_buffer(self):

        """
         :return:
        """

        self.weight_buffer = pd.DataFrame(index=self.features.index,columns=self.asset_names).fillna(0)+ 1 / self.number_of_assets

    @property
    def shape(self):
        raise

    def _set_weights_on_date(self,weights, target_date):
        self.weight_buffer.loc[target_date] = weights

    def step(self, action, action_date):
        """

        :param action: corresponds to portfolio weights np.array(n_assets,1)
        :param action_date: datetime.datetime
        :return:
        """
        # get previous allocation

        action_date_index = self.weight_buffer.index.searchsorted(action_date)

        next_observation_date = self.forward_returns_dates.iloc[action_date_index].values[0]
        next_observation_date_index = self.weight_buffer.index.searchsorted(next_observation_date)
        # on each step between  action_date and next observation date , the weights should be refilled
        self.weight_buffer.iloc[action_date_index:next_observation_date_index,:]=action


        weight_difference = self.weight_buffer.iloc[action_date_index - 1:action_date_index + 1]
        # obtain the difference from the previous allocation, diff is done t_1 - t
        weight_difference = abs(weight_difference.diff().dropna())

        # calculate rebalance commission
        commision_percent_cost = -weight_difference.sum(axis = 1) * self.percent_commission

        # get period_ahead_returns
        t_plus_one_returns = self.forward_returns.iloc[action_date_index]
        one_period_mtm_reward = (t_plus_one_returns * action).sum()

        reward = one_period_mtm_reward - commision_percent_cost





        done= False
        extra_info={"action_date":action_date,
            "forward_returns":t_plus_one_returns,
                    "previous_weights":self.weight_buffer.iloc[action_date_index - 1]}
        return next_observation_date,reward,done,extra_info

    def encode(self, date):
        """
        convert current state to tensor

        """
        pass

    def get_state_on_date(self, target_date):
        """
            returns the state on a target date
           :param target_date:
           :return: in_window_features, weights_on_date
        """
        #TODO: what happens for  different features for example ("Non Time Series Returns")?
        assert target_date >= self.features.index[self.in_bars_count]

        date_index = self.features.index.searchsorted(target_date)
        state_features =self.features.iloc[date_index - self.in_bars_count + 1:date_index + 1]
        weights_on_date = self.weight_buffer.iloc[date_index]

        return state_features, weights_on_date


class DeepTradingEnvironment(gym.Env):
    metadata = {'render.modes': ['human']}
    RESAMPLE_DATA_FREQUENCY="5min"

    @staticmethod
    def _build_and_persist_features(assets_prices, out_reward_window,data_hash):
        """
         builds close-to-close returns for a specif
         :param self:
         :param assets_prices:(DataFrame)
         :param out_reward_window:(datetime.timedelta)
         :return:
         """

        PERSISTED_DATA_DIRECTORY = "temp_persisted_data"
        # Todo: Hash csv file
        if not os.path.exists(PERSISTED_DATA_DIRECTORY + "/only_features_"+data_hash):
            features_instance=DailyDataFrame2Features(bars_dict={col:assets_prices[col] for col in assets_prices.columns}
                                                      ,configuration_dict={},
                                                      forward_returns_time_delta=[out_reward_window])

            features=features_instance.all_features

            only_features, only_forward_returns =features_instance.separate_features_from_forward_returns(features=features)

            #Todo: get all features
            only_features=only_features[[col for col in only_features.columns if "log_return" in col]]
            forward_returns_dates = features_instance.forward_returns_dates[0]

            only_features.to_parquet(PERSISTED_DATA_DIRECTORY + "/only_features_" + data_hash)
            only_forward_returns.to_parquet(PERSISTED_DATA_DIRECTORY + "/only_forward_returns_" + data_hash)
            forward_returns_dates.to_parquet(PERSISTED_DATA_DIRECTORY + "/forward_return_dates_" + data_hash)

        else:

            only_features = pd.read_parquet(PERSISTED_DATA_DIRECTORY + "/only_features_"+data_hash)
            only_forward_returns=pd.read_parquet(PERSISTED_DATA_DIRECTORY + "/only_forward_returns_"+data_hash)
            forward_returns_dates=pd.read_parquet(PERSISTED_DATA_DIRECTORY + "/forward_return_dates_" + data_hash)


        return only_features, only_forward_returns,forward_returns_dates


    @classmethod
    def build_environment_from_simulated_assets(cls,assets_simulation_details,data_hash,
                                                meta_parameters,objective_parameters,periods=1000):
        """
        Simulates continous 1 minute data
        :param assets_simulation_details: (dict)
        :param simulation_details: (dict)
        :param meta_parameters: (dict)
        :param objective_parameters: (dict)
        :param periods:
        :param simulation_method:
        :return: DeepTradingEnvironment
        """


        date_range=pd.date_range(start=datetime.datetime.utcnow(),periods=periods,freq="1d") #change period to 1Min
        asset_prices=pd.DataFrame(index=date_range,columns=list(assets_simulation_details.keys()))
        for asset,simulation_details in assets_simulation_details.items():
            new_asset=SimulatedAsset()
            #time in years in minutes=1/(252*570)
            asset_prices[asset]=new_asset.simulate_returns(time_in_years=1/(252),n_returns=periods,**simulation_details)

        asset_prices=asset_prices.cumprod()

        return cls._create_environment_from_asset_prices(assets_prices=asset_prices,data_hash=data_hash,
                                                         meta_parameters=meta_parameters,objective_parameters=objective_parameters)
    @classmethod
    def _create_environment_from_asset_prices(cls,assets_prices,meta_parameters,objective_parameters,data_hash,*args,**kwargs):
        """

        :param assets_prices:  (pandas.DataFrame)
        :return: DeepTradingEnvironment
        """

        # resample
        assets_prices = assets_prices.resample(cls.RESAMPLE_DATA_FREQUENCY).first()
        assets_prices = assets_prices.dropna()
        features, forward_returns,forward_returns_dates = cls._build_and_persist_features(assets_prices=assets_prices,
                                             out_reward_window=meta_parameters["out_reward_window"],
                                             data_hash=data_hash)


        # transform features

        return DeepTradingEnvironment(features=features,forward_returns_dates=forward_returns_dates,
                               forward_returns=forward_returns, meta_parameters=meta_parameters,
                               objective_parameters=objective_parameters)

    @classmethod
    def build_environment_from_dirs_and_transform(cls, meta_parameters, objective_parameters,data_hash, data_dir="data_env", **kwargs):
        """
        Do transformations that shouldnt be part of the class

        Also using the meta parameters


        """
        # optimally this should be only features
        assets_prices = {file: pd.read_parquet(data_dir + "/" + file)["close"] for file in os.listdir(data_dir)}
        assets_prices = pd.DataFrame(assets_prices)

        environment=cls._create_environment_from_asset_prices(assets_prices=assets_prices,data_hash=data_hash,
                                                              meta_parameters=meta_parameters,objective_parameters=objective_parameters)

        return environment

    def __init__(self, features, forward_returns,forward_returns_dates, objective_parameters,
                 meta_parameters):
        """
          features and forward returns should be aligned by the time axis. The setup should resemble a supervised learning

          :param features: pandas.DataFrame, historical features
          :param forward_returns: pandas.DataFrame, assets forward returns
          :param objective_parameters:
          :param meta_parameters:
        """

        assert features.index.equals(forward_returns.index)

        self.features = features
        self.forward_returns = forward_returns
        self.forward_returns_dates=forward_returns_dates
        # create helper variables
        self._set_environment_helpers()
        self._set_reward_helpers(objective_parameters)

        self._set_state(meta_parameters=meta_parameters,objective_parameters=objective_parameters)


        # action space is the portfolio weights at any time in our example it is bounded by [0,1]
        self.action_space = gym.spaces.Box(low=0, high=1, shape=(self.number_of_assets,))
        # features to be scaled normal scaler will bound them in -4,4
        self.observation_space = gym.spaces.Box(low=-4, high=4, shape=(self.number_of_features,))

    def _set_state(self,meta_parameters,objective_parameters):
        # logic to create state
        state_type=meta_parameters["state_type"]
        if state_type =="in_window_out_window":
            # Will be good if meta parameters does not need to be passed even to the environment possible?
            self.state = State(features=self.features,
                                in_bars_count=meta_parameters["in_bars_count"],
                                objective_parameters=objective_parameters,
                               forward_returns=self.forward_returns,
                               forward_returns_dates=self.forward_returns_dates

                                )


    def _set_reward_helpers(self,objective_parameters):
        # case for interval return
        self.objective_parameters = objective_parameters

    def _set_environment_helpers(self):
        """
        creates helper variables for the environment
        """
        self.number_of_assets = len(self.forward_returns.columns)
        self.number_of_features=len(self.features)


    def reset(self):
        """
        resets the environment:
            -resets the buffer of weights in the environments

        """

    def step(self, action_portfolio_weights, action_date):
        """

        :param action_portfolio_weights:
        :param action_date:
        :return:
        """

        action = action_portfolio_weights
        observation,reward,done,extra_info= self.state.step(action, action_date)
        # obs = self._state.encode()
        obs=observation
        info=extra_info
        return obs, reward, done, info

    def render(self, mode='human', close=False):
        pass

    def close(self):
        pass

import math
def sigmoid(x):
        return 1 / (1 + math.exp(-x))

class LinearAgent:

    def __init__(self,environment,out_reward_window_td):
        """



        :param environment:
        :param out_reward_window_td: datetime.timedelta,
        """
        self.environment = environment
        self.out_reward_window_td=out_reward_window_td
        self._initialize_helper_properties()
        self._initialize_linear_parameters()

    def _initialize_helper_properties(self):

        self.number_of_assets=self.environment.number_of_assets
        self.state_features_shape=self.environment.state.state_features_shape
    def _initialize_linear_parameters(self):
        """
        parameters are for mu and sigma
        (features_rows*features_columns +number_of_assets(weights))*number of asssets
        :return:
        """


        param_dim=(self.state_features_shape[0]\
                                   *self.state_features_shape[1]+self.number_of_assets)*self.number_of_assets
        self.theta_mu=np.random.rand(param_dim)
        self.theta_sigma=np.random.rand(param_dim)


    def _get_mus(self,theta_mu):

        theta_mu_features=theta_mu[:-self.number_of_assets*self.number_of_assets].reshape(self.number_of_assets,
                                                                    self.state_features_shape[0],self.state_features_shape[1])
        theta_mu_weights=theta_mu[-self.number_of_assets*self.number_of_assets:].reshape(self.number_of_assets,
                                                                                         self.number_of_assets)


        return theta_mu_features, theta_mu_weights

    def _get_sigmas(self,theta_sigma):

        return  self._get_mus(theta_sigma)

    def _policy_linear(self,state_features,weights_on_date):
        """
        return action give a linear policy
        :param state:
        :param action_date:
        :return:
        """
        theta_mu_features, theta_mu_weights=self._get_mus(self.theta_mu)
        theta_sigma_features, theta_sigma_weights = self._get_mus(self.theta_sigma)
        mu_features=(theta_mu_features*state_features.values)
        mu_features=mu_features.sum(axis=1).sum(axis=1)
        mu_weights=(theta_mu_weights*weights_on_date.values).sum(axis=1)
        mu=mu_features+mu_weights

        sigma_features = (theta_sigma_features * state_features.values)
        sigma_features = sigma_features.sum(axis=1).sum(axis=1)
        sigma_weights = (theta_sigma_weights * weights_on_date.values).sum(axis=1)
        #guarantee is always positive
        sigma_total = np.exp(sigma_features + sigma_weights)
        # stabilize values
        mu = np.array(list(map(sigmoid, mu)))
        sigma_squared=np.array(list(map(sigmoid, sigma_total**2)))


        cov=np.zeros((self.number_of_assets,self.number_of_assets))
        np.fill_diagonal(cov,sigma_squared)

        action=np.random.multivariate_normal(mu,cov)

        return action


    def get_best_action(self,state,action_date):
        """
        returns best action given state (portfolio weights
        :param state:
        :param action_date:
        :return:
        """
        state_features, weights_on_date = state.get_state_on_date(target_date=action_date)

        action=self._policy_linear(state_features=state_features,weights_on_date=weights_on_date)


        return action

    def sample_env(self,observations=100):
        start = np.random.choice(range(self.environment.state.in_bars_count,self.environment.features.shape[0]))
        start_date =self.environment.features.index[start]
        rewards = []

        for counter,iloc_date in enumerate(range(start, start + observations, 1)):
            if counter==0:
                action_date=start_date


            action_portfolio_weights = self.get_best_action(self.environment.state,action_date=action_date)
            action_date, reward, done, info = self.environment.step(action_portfolio_weights=action_portfolio_weights,
                                                action_date=action_date)
            print(info)

            rewards.append(reward)
