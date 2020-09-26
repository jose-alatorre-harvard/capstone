import gym
import pandas as pd
import numpy as np
import os
import datetime
from tqdm import tqdm
from lib.Benchmarks import SimulatedAsset


class State:

    def __init__(self, features, in_bars_count, objective_parameters):
        """

        :param features: (dict)
        :param in_bars_count: (int)
        :param objective_parameters:(dict)

        """

        self.features = features
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

        self.log_close_returns = self.features["log_close_returns"]
        self.assets_names=self.log_close_returns.columns
        self.number_of_assets=len(self.assets_names)

        self.state_features_shape=(self.in_bars_count,self.features["input_features"].shape[1])

    def _set_objective_function_parameters(self,objective_parameters):
        self.percent_commission = objective_parameters["percent_commission"]

    def reset(self):
        """
        resets the weights_buffer

        """

        self._initialize_weights_buffer()

    def _initialize_weights_buffer(self):
        # TODO: Should this be part of state or environment?
        """
         :return:
        """

        self.weight_buffer = pd.DataFrame(index=self.log_close_returns.index,columns=self.assets_names).fillna(0)+ 1 / self.number_of_assets

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

        action_date_index = np.argmax(self.weight_buffer.index.isin([action_date]))
        self._set_weights_on_date(weights=action, target_date=action_date)

        weight_difference = self.weight_buffer.iloc[action_date_index - 1:action_date_index + 1]
        # obtain the difference from the previous allocation, diff is done t_1 - t
        weight_difference = abs(weight_difference.diff().dropna())

        # calculate rebalance commission
        commision_percent_cost = -weight_difference.sum(axis = 1) * self.percent_commission

        # get period_ahead_returns
        t_plus_one_returns = self.log_close_returns.iloc[action_date_index]
        one_period_mtm_reward = (t_plus_one_returns * action).sum()

        reward = one_period_mtm_reward - commision_percent_cost
        observation=t_plus_one_returns.values
        done= False
        extra_info={}
        return observation,reward,done,extra_info

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
        assert target_date >= self.features["input_features"].index[self.in_bars_count]

        date_index = self.features["input_features"].index.searchsorted(target_date)
        state_features =self.features["input_features"].iloc[date_index - self.in_bars_count + 1:date_index + 1]
        weights_on_date = self.weight_buffer.iloc[date_index]

        return state_features, weights_on_date


class DeepTradingEnvironment(gym.Env):
    metadata = {'render.modes': ['human']}
    RESAMPLE_DATA_FREQUENCY="5min"

    @staticmethod
    def _buid_close_returns(assets_prices, out_reward_window,data_hash):
        """
         builds close-to-close returns for a specif
         :param self:
         :param assets_prices:(DataFrame)
         :param out_reward_window:(datetime.timedelta)
         :return:
         """

        PERSISTED_DATA_DIRECTORY = "temp_persisted_data"
        # Todo: Hash csv file
        if not os.path.exists(PERSISTED_DATA_DIRECTORY + "/log_close_returns_"+data_hash):
            try:
                log_close_returns = assets_prices.copy() * np.nan
                cross_over_day = log_close_returns[log_close_returns.columns[0]].fillna(0)
                next_return_time_stamp = log_close_returns[log_close_returns.columns[0]].fillna(0)
                for counter, date in tqdm(enumerate(assets_prices.index)):
                    next_date = date + out_reward_window

                    next_date_index = assets_prices.index.searchsorted(next_date)
                    log_close_returns.iloc[counter] = np.log(
                        assets_prices.iloc[next_date_index] / assets_prices.iloc[counter])

                    next_return_time_stamp.iloc[counter] = assets_prices.index[next_date_index]
                    if assets_prices.index[next_date_index].date() != assets_prices.index[counter].date():
                        # cross over date
                        cross_over_day.iloc[counter] = 1


            except:
                log_close_returns.iloc[counter] = np.nan

            log_close_returns.to_parquet(PERSISTED_DATA_DIRECTORY + "/log_close_returns_"+data_hash)
            cross_over_day = pd.DataFrame(cross_over_day)
            cross_over_day.to_parquet(PERSISTED_DATA_DIRECTORY + "/cross_over_day_"+data_hash)
            next_return_time_stamp = pd.DataFrame(next_return_time_stamp)
            next_return_time_stamp.to_parquet(PERSISTED_DATA_DIRECTORY + "/next_return_time_stamp_"+data_hash)
            assets_prices.to_parquet(PERSISTED_DATA_DIRECTORY + "/assets_prices"+data_hash)
        else:

            log_close_returns = pd.read_parquet(PERSISTED_DATA_DIRECTORY + "/log_close_returns_"+data_hash)
            cross_over_day = pd.read_parquet(PERSISTED_DATA_DIRECTORY + "/cross_over_day_"+data_hash)
            next_return_time_stamp = pd.read_parquet(PERSISTED_DATA_DIRECTORY + "/next_return_time_stamp_"+data_hash)
            assets_prices= pd.read_parquet(PERSISTED_DATA_DIRECTORY + "/assets_prices"+data_hash)

        return {"input_features": log_close_returns,"assets_prices":assets_prices,
                "log_close_returns": log_close_returns, "cross_over_day": cross_over_day,
                "next_return_time_stamp": next_return_time_stamp}

    @classmethod
    def build_environment_from_simulated_assets(cls,assets_simulation_details,data_hash,
                                                meta_parameters,objective_parameters,periods=100000):
        """
        Simulates Continous 1 minute data
        :param assets_simulation_details: (dict)
        :param simulation_details: (dict)
        :param meta_parameters: (dict)
        :param objective_parameters: (dict)
        :param periods:
        :param simulation_method:
        :return: DeepTradingEnvironment
        """


        date_range=pd.date_range(start=datetime.datetime.utcnow(),periods=periods,freq="1min")
        asset_prices=pd.DataFrame(index=date_range,columns=list(assets_simulation_details.keys()))
        for asset,simulation_details in assets_simulation_details.items():
            new_asset=SimulatedAsset()
            asset_prices[asset]=new_asset.simulate_returns(time_in_years=1/(252*570),n_returns=periods,**simulation_details)

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
        input_data = cls._buid_close_returns(assets_prices=assets_prices,
                                             out_reward_window=meta_parameters["out_reward_window"],
                                             data_hash=data_hash)

        #rewrite asset prices for data consistency
        assets_prices=input_data["assets_prices"]

        # transform features

        return DeepTradingEnvironment(input_data=input_data,
                               assets_prices=assets_prices, meta_parameters=meta_parameters,
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

    def __init__(self, input_data, assets_prices, objective_parameters,
                 meta_parameters):
        """
        features: pandas.DataFrame with features by time
        asset_prices=pandas.DataFrame with asset prices by time
        """

        assert input_data["log_close_returns"].index.equals(assets_prices.index)
        assert input_data["input_features"].index.equals(assets_prices.index)
        assert input_data["cross_over_day"].index.equals(assets_prices.index)

        self.features = input_data
        self.assets_prices = assets_prices
        # create helper variables
        self._set_environment_helpers()
        self._set_reward_helpers(assets_prices, objective_parameters, meta_parameters)

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

                                )


    def _set_reward_helpers(self, assets_prices, objective_parameters, meta_parameters):
        # case for interval return
        self.objective_parameters = objective_parameters

    def _set_environment_helpers(self):
        """
        creates helper variables for the environment
        """
        self.number_of_assets = len(self.assets_prices.columns)
        self.number_of_features=len(self.features["input_features"])


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
        info = {"action": action,
                "date": action_date}

        return obs, reward, done, info

    def render(self, mode='human', close=False):
        pass

    def close(self):
        pass

import math
def sigmoid(x):
        return 1 / (1 + math.exp(-x))

class LinearAgent:

    def __init__(self,environment):
        self.environment = environment
        self._initialize_helper_properties()
        self._initialize_linear_parameters()

    def _initialize_helper_properties(self):

        self.number_of_assets=len(self.environment.assets_prices.columns)
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
        log_sigma_squared = np.exp(sigma_features + sigma_weights)
        # stabilize values
        mu = np.array(list(map(sigmoid, mu)))
        sigma_squared=np.array(list(map(sigmoid, np.log(log_sigma_squared))))


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


        return np.random.rand(2)

    def sample_env(self,observations=100):
        start = np.random.choice(range(self.environment.assets_prices.shape[0]))

        rewards = []
        # Todo: Note that sum of rewards assume that we rebalance a bet on each step ok for training?
        for iloc_date in range(start, start + observations, 1):
            action_date = self.environment.assets_prices.index[iloc_date]
            action_portfolio_weights = self.get_best_action(self.environment.state,action_date=action_date)
            obs, reward, done, info = self.environment.step(action_portfolio_weights=action_portfolio_weights,
                                                action_date=action_date)

            rewards.append(reward)

