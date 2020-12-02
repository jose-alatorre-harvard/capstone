import gym
import pandas as pd
import numpy as np
import os
import datetime
from tqdm import tqdm
from lib.Benchmarks import SimulatedAsset

import matplotlib.pyplot as plt

import copy
import warnings
from joblib import Parallel, delayed



# from tensorflow import keras
# from tensorflow.keras import layers
# from tensorflow.keras.layers.experimental import preprocessing
# import tensorflow as tf
# from tensorflow.keras import backend as K
# import tensorflow.keras.initializers as initializers
from utils import DailyDataFrame2Features
import torch
import torch.nn.functional as F
class RewardFactory:

    def __init__(self,in_bars_count,percent_commission):


        self.in_bars_count=in_bars_count
        self.percent_commission=percent_commission


    def get_reward(self, weights_bufffer,forward_returns,action_date_index,reward_function):
        """
        launch reward types Needs to be implemented
        :param reward:
        :return:
        """
        portfolio_returns=self._calculate_returns_with_commisions( weights_bufffer, forward_returns, action_date_index)

        if reward_function == "cum_return":
            return self._reward_cum_return(portfolio_returns)
        elif reward_function == "max_sharpe":
            return self._reward_max_sharpe(portfolio_returns)
        elif reward_function == "min_vol":
            return self._reward_to_min_vol(portfolio_returns)

    def _reward_to_min_vol(self, portfolio_returns):
        """
        minimum volatility portfolio
        :param period_returns:
        :return:
        """


        return -portfolio_returns.std()*np.sqrt(252 / 7)

    def _reward_max_sharpe(self, portfolio_returns):
        """
        calculates sharpe ratio for the returns
        :param period_returns:
        :return:
        """


        mean_return = portfolio_returns.mean() * (252 / 7)
        vol = portfolio_returns.std() * np.sqrt(252 / 7)
        sharpe = mean_return / (vol)


        return sharpe

    def _reward_cum_return(self, portfolio_returns):

        return portfolio_returns.iloc[-1]

    def _calculate_returns_with_commisions(self,weights_buffer,forward_returns,action_date_index):
        """
        calculates the effective returns with commision
        :param target_weights:
        :return:
        """
        target_weights=weights_buffer.iloc[action_date_index -self.in_bars_count- 1:action_date_index + 1]
        target_forward_returns=forward_returns.iloc[action_date_index -self.in_bars_count- 1:action_date_index + 1]

        weight_difference = abs(target_weights.diff())
        commision_percent_cost = -weight_difference.sum(axis=1) * self.percent_commission

        portfolio_returns=(target_forward_returns*target_weights).sum(axis=1)-commision_percent_cost

        return portfolio_returns
class State:

    def __init__(self, features,forward_returns,asset_names,in_bars_count,forward_returns_dates, objective_parameters,
                 include_previous_weights=True):
        """

          :param features:
          :param forward_returns:
          :param forward_returns_dates:
          :param objective_parameters:
        """

        self.features = features
        self.a_names=asset_names
        self.forward_returns=forward_returns
        self.forward_returns.columns=self.asset_names
        self.include_previous_weights=include_previous_weights
        self.forward_returns_dates=forward_returns_dates
        self.in_bars_count=in_bars_count
        self._set_helper_functions()
        self._set_objective_function_parameters(objective_parameters)

        self._initialize_weights_buffer()
        self.reward_factory=RewardFactory(in_bars_count=in_bars_count,percent_commission=self.percent_commission)

    def get_flat_state_by_iloc(self,index_location):
        """

        :return:
        """

        state_features,weights_on_date=self.get_state_by_iloc(index_location=index_location)
        return self.flatten_state(state_features, weights_on_date)

    def flatten_state(self,state_features, weights_on_date):
        """
        flatten states by adding weights to features

        :return:
        """
        flat_state=state_features.copy()

        # for index in weights_on_date.index:
        #     flat_state[index] = weights_on_date.loc[index]
        if self.include_previous_weights==True:
            flat_state=pd.concat([flat_state,weights_on_date],axis=0)
        else:
            flat_state=flat_state
        return flat_state

    def _set_helper_functions(self):
        """
        Creates following properties
        assets_names: (list)
        log_close_returns: (pd.DataFrame)
        :return:
        """

        self.number_of_assets=len(self.forward_returns.columns)
        if self.include_previous_weights==True:
            self.state_dimension=self.features.shape[1] +self.number_of_assets#*self.in_bars_count
        else:
            self.state_dimension=self.features.shape[1]

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
        if self.a_names==None:

            return self.forward_returns.columns
        else:
            return self.a_names

    def _initialize_weights_buffer(self):

        """
         :return:
        """
        #initialise weights uniform
        init_w=np.random.uniform(0,1,(len(self.features.index),len(self.asset_names)))
        init_w=np.apply_along_axis(lambda x: np.exp(x) / np.sum(np.exp(x)), 1, init_w)
        self.weight_buffer = pd.DataFrame(index=self.features.index,columns=self.asset_names,
                                          data=init_w)


    @property
    def shape(self):
        raise

    def _set_weights_on_date(self,weights, target_date):
        self.weight_buffer.loc[target_date] = weights

    def sample_rewards_by_indices(self,sample_indices,reward_function):
        """

        :param sample_indices:
        :return:
        """
        rewards = []
        for i in sample_indices:
            reward = self.reward_factory.get_reward(weights_bufffer=self.weight_buffer,
                                                    forward_returns=self.forward_returns,
                                                    action_date_index= i,
                                                    reward_function=reward_function)
            rewards.append(reward)

        return rewards


    def sample_rewards(self,action_date_index,sample_size,reward_function):
        """

        :return:
        """
        rewards=[]
        for i in range(sample_size):

            reward = self.reward_factory.get_reward(weights_bufffer=self.weight_buffer,
                                                    forward_returns=self.forward_returns,
                                                    action_date_index=action_date_index+i,
                                                    reward_function=reward_function)
            rewards.append(reward)

        return rewards

    def sample_state_by_iloc(self, index_location, sample_size):
        """

        :param index_location:
        :param sample_size:
        :return:
        """
        states = []
        for i in range(sample_size):
            state = self.get_flat_state_by_iloc(index_locatifon=index_location + 1)
            states.append(state.values)
        return states

    def update_weights_by_iloc(self,index_location,sample_size,new_weights):
        """

        :param index_location:
        :param sample_size:
        :return:
        """
        self.weight_buffer.iloc[index_location:index_location+sample_size]=new_weights

    def step(self, action, action_date,reward_function,pre_indices=None):
        """

        :param action: corresponds to portfolio weights np.array(n_assets,1)
        :param action_date: datetime.datetime
        :return:
        """
        # get previous allocation
        if pre_indices is not None:
            action_date_index=pre_indices[0]
            next_observation_date_index=pre_indices[1]
            next_observation_date= self.forward_returns_dates.iloc[action_date_index].values[0]
        else:
            action_date_index = self.weight_buffer.index.searchsorted(action_date)
            next_observation_date = self.forward_returns_dates.iloc[action_date_index].values[0]
            next_observation_date_index = self.weight_buffer.index.searchsorted(next_observation_date)

        # on each step between  action_date and next observation date , the weights should be refilled

        self.weight_buffer.iloc[action_date_index:next_observation_date_index,:]=action

        reward=self.reward_factory.get_reward(weights_bufffer=self.weight_buffer,
                                              forward_returns=self.forward_returns,
                                              action_date_index=action_date_index,
                                              reward_function=reward_function)


        #reward factory_should be launched_here



        done= False
        extra_info={"action_date":action_date,
            "reward_function":reward_function,
                    "previous_weights":self.weight_buffer.iloc[action_date_index - 1]}
        return next_observation_date,reward,done,extra_info

    def encode(self, date):
        """
        convert current state to tensor

        """
        pass
    def get_full_state_pre_process(self):
        """
        gets full state data
        :return:
        """
        state_features = self.features
        weights_on_date = self.weight_buffer.applymap(lambda x : np.random.rand())
        return pd.concat([state_features,weights_on_date],axis=1)

    def get_state_by_iloc(self,index_location):
        """
        index location
        :param iloc:
        :return:
        """

        state_features = self.features.iloc[index_location]
        weights_on_date = self.weight_buffer.iloc[index_location]

        return state_features,weights_on_date



    def get_state_on_date(self, target_date,pre_indices=None):
        """
            returns the state on a target date
           :param target_date:
           :return: in_window_features, weights_on_date
        """
        #TODO: what happens for  different features for example ("Non Time Series Returns")?
        try:
            assert target_date >= self.features.index[0]
            if pre_indices is None:
                date_index = self.features.index.searchsorted(target_date)
            else:
                date_index=pre_indices[0]

            state_features, weights_on_date=self.get_state_by_iloc(index_location=date_index)

        except:
            raise
        return state_features, weights_on_date





class DeepTradingEnvironment(gym.Env):
    metadata = {'render.modes': ['human']}


    @staticmethod
    def _build_and_persist_features(assets_dict, out_reward_window,in_bars_count,data_hash):
        """
         builds close-to-close returns for a specif
        :param assets_dict:
        :param out_reward_window:
        :param in_bars_count:
        :param data_hash:
        :return:
        """



        PERSISTED_DATA_DIRECTORY = "temp_persisted_data"
        # Todo: Hash csv file
        if not os.path.exists(PERSISTED_DATA_DIRECTORY + "/only_features_"+data_hash):
            features_instance=DailyDataFrame2Features(bars_dict=assets_dict
                                                      ,configuration_dict={},
                                                      forward_returns_time_delta=[out_reward_window])

            features=features_instance.all_features

            only_features, only_forward_returns =features_instance.separate_features_from_forward_returns(features=features)
            forward_returns_dates = features_instance.forward_returns_dates
            #Todo: get all features
            only_features=only_features[[col for col in only_features.columns if "log_return" in col]]
            #get the lagged returns as features
            only_features=features_instance.add_lags_to_features(only_features,n_lags=in_bars_count)
            only_features=only_features.dropna()
            only_forward_returns=only_forward_returns.reindex(only_features.index)
            forward_returns_dates=forward_returns_dates.reindex(only_features.index)
            #




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
                                                meta_parameters,objective_parameters,periods=2000):
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


        date_range=pd.date_range(start=datetime.datetime.utcnow(),periods=periods,freq="1d",normalize=True) #change period to 1Min
        asset_prices=pd.DataFrame(index=date_range,columns=list(assets_simulation_details.keys()))
        for asset,simulation_details in assets_simulation_details.items():
            new_asset=SimulatedAsset()
            #time in years in minutes=1/(252*570)
            asset_prices[asset]=new_asset.simulate_returns(time_in_years=1/(252),n_returns=periods,**simulation_details)

        asset_prices=asset_prices.cumprod()
        assets_dict={col :asset_prices[col] for col in asset_prices.columns}

        return cls._create_environment_from_assets_dict(assets_dict=assets_dict,data_hash=data_hash,
                                                         meta_parameters=meta_parameters,objective_parameters=objective_parameters)
    @classmethod
    def _create_environment_from_assets_dict(cls,assets_dict,meta_parameters,objective_parameters,data_hash,*args,**kwargs):
        """

        :param assets_prices:  (pandas.DataFrame)
        :return: DeepTradingEnvironment
        """

        # resample
        features, forward_returns,forward_returns_dates = cls._build_and_persist_features(assets_dict=assets_dict,
                                                                    in_bars_count=meta_parameters["in_bars_count"],
                                             out_reward_window=meta_parameters["out_reward_window"],
                                             data_hash=data_hash)


        # add bias to features
        features["bias"]=1

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
        #RESAMPLE NEEDS RE-CREATION OF TIME SERIE SO JUST USE THIS FOR TESTING
        assets_dict = {file: pd.read_parquet(data_dir + "/" + file).resample("30min").first() for file in os.listdir(data_dir)}
        counter=0
        for key, value in assets_dict.items():
            if counter==0:
                main_index=value.index
            else:
                main_index=main_index.join(value.index,how="inner")

        for key, value in assets_dict.items():
            tmp_df=value.reindex(main_index)
            tmp_df=tmp_df.fillna(method='ffill')
            assets_dict[key]=tmp_df


        environment=cls._create_environment_from_assets_dict(assets_dict=assets_dict,data_hash=data_hash,
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
        if "asset_names" in meta_parameters:
            asset_names=meta_parameters["asset_names"]
        else:
            asset_names=None
        if state_type =="in_window_out_window":
            # Will be good if meta parameters does not need to be passed even to the environment possible?
            self.state = State(features=self.features,
                               asset_names=asset_names,
                               in_bars_count=meta_parameters["in_bars_count"],
                               objective_parameters=objective_parameters,
                               forward_returns=self.forward_returns,
                               forward_returns_dates=self.forward_returns_dates,
                               include_previous_weights=meta_parameters["include_previous_weights"]

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

    def step(self, action_portfolio_weights, action_date,reward_function,pre_indices=None):
        """

        :param action_portfolio_weights:
        :param action_date:
        :return:
        """

        action = action_portfolio_weights
        observation,reward,done,extra_info= self.state.step(action=action,
                                                            action_date=action_date,
                                                            reward_function=reward_function,
                                                            pre_indices=pre_indices)
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




class AgentDataBase:

    def __init__(self, environment, out_reward_window_td, reward_function, sample_observations=32,pre_sample=True):
        self.environment = environment
        self.out_reward_window_td = out_reward_window_td
        self.sample_observations = sample_observations
        self.reward_function = reward_function
        self._initialize_helper_properties()
        if pre_sample==True:
            # self._build_full_pre_sampled_indices()
            self._set_latest_posible_date()

    def _initialize_helper_properties(self):
        self.number_of_assets = self.environment.number_of_assets
        self.state_dimension = self.environment.state.state_dimension

    def backtest_policy(self,epoch,backtest):
        """
        backtest portfolio policy
        :return:
        """
        activate_date=self.environment.features.index[0]
        tmp_weights=self.environment.state.weight_buffer.copy()

        fwd_return_date_name=self.environment.forward_returns_dates.columns[0]
        while activate_date <= self.environment.forward_returns_dates.iloc[-10].values[0]:

            i=self.environment.features.index.searchsorted(activate_date)
            try:
                obs=self.environment.state.get_flat_state_by_iloc(i)
            except:
                a=5
            action=self.policy(obs,deterministic=True)
            next_observation_date=self.environment.forward_returns_dates.iloc[i][fwd_return_date_name]

            tmp_weights.loc[activate_date:next_observation_date,:]=action
            activate_date=next_observation_date

        tmp_backtest=((self.environment.forward_returns*tmp_weights).sum(axis=1)+1).cumprod()
        tmp_backtest.name=epoch
        if backtest is None:
            backtest=tmp_backtest.to_frame()
        else:
            backtest=pd.concat([backtest,tmp_backtest],axis=1)

        return backtest
    def _build_full_pre_sampled_indices(self):
        """
        build full pre sampled indices
        :return:
        """

        forward_return_dates_df=self.environment.forward_returns_dates
        first_forward_return_date=forward_return_dates_df.iloc[self.environment.state.in_bars_count+1].values[0]

        sample_indices={}
        date_start_counter=self.environment.state.in_bars_count+1
        start_date = forward_return_dates_df.index[date_start_counter]
        while start_date != first_forward_return_date:

            start_date = forward_return_dates_df.index[date_start_counter]
            sample_indices[start_date]=[date_start_counter]

            index_date_iloc=date_start_counter
            while index_date_iloc <forward_return_dates_df.shape[0]:

                forward_return_date=forward_return_dates_df.iloc[index_date_iloc].values[0]
                forward_return_date_index=forward_return_dates_df.index.searchsorted(forward_return_date)
                if forward_return_date <forward_return_dates_df.index[-1]:
                    sample_indices[start_date].append(forward_return_date_index)

                index_date_iloc=forward_return_date_index

            date_start_counter=date_start_counter+1

        self.sample_indices=sample_indices
    def _set_latest_posible_date(self):
        """

        :param observations:
        :return:
        """

        frd=self.environment.forward_returns_dates
        column_name = frd.columns[0]
        end_date=frd[column_name].max()


        for obs in range(self.sample_observations+1):
            last_date_start = frd[frd[column_name] == end_date].index
            last_date_start_index = frd.index.searchsorted(last_date_start)
            end_date = frd.index[last_date_start_index][0]
        self.latest_posible_index_date=last_date_start_index[0]
        self.max_available_obs_date=frd[column_name].index.max().tz_localize(None)

        # presampled indices for environment sample

        self.pre_sample_date_indices = pd.DataFrame(index=self.environment.forward_returns_dates.index,
                                                    columns=range(self.sample_observations+1))
        # todo assert forward return dates index, equals buffer

        for iloc in tqdm(range(self.latest_posible_index_date),
                         desc="pre-sampling indices"):

            start_date = self.environment.forward_returns_dates.index[iloc]
            nfd = self.environment.forward_returns_dates
            indices = []
            for obs in range(self.sample_observations+1):

                if obs == 0:
                    start_date_index = iloc
                else:
                    start_date_index = nfd.index.searchsorted(next_date)
                indices.append(start_date_index)
                next_date = nfd.iloc[start_date_index][nfd.columns[0]]

            self.pre_sample_date_indices.loc[start_date, :] = indices

    def get_best_action(self,flat_state):
        """
        returns best action given state (portfolio weights
        :param state:
        :param action_date:
        :return:
        """

        action=self.policy(flat_state=flat_state)


        return action

    def policy(self,flat_state,deterministic=False):
        raise NotImplementedError
    def _get_sars_by_date(self,action_date,verbose=False,pre_indices=None):
        """
        gets sars by date
        :param action_date:
        :return:
        """
        state_features, weights_on_date = self.environment.state.get_state_on_date(target_date=action_date,
                                                                                   pre_indices=pre_indices)
        flat_state = self.environment.state.flatten_state(state_features=state_features,
                                                          weights_on_date=weights_on_date)
        action_portfolio_weights = self.get_best_action(flat_state=flat_state)

        next_action_date, reward, done, info = self.environment.step(
            action_portfolio_weights=action_portfolio_weights,reward_function=self.reward_function,
            action_date=action_date,pre_indices=pre_indices)

        if verbose:

            print(info)

        return next_action_date,  flat_state ,reward, action_portfolio_weights

    def sample_env_pre_sampled(self,verbose=False):
        # starts in 1 becasue comission dependes on initial weights
        start = np.random.choice(range(self.environment.state.in_bars_count + 1, self.latest_posible_index_date))
        states, actions, rewards = self.sample_env_pre_sampled_from_index(start=start,
                                                                            sample_observations=self.sample_observations,
                                                                          pre_sample_date_indices=self.pre_sample_date_indices ,
                                                                          forward_returns_dates=self.environment.forward_returns_dates)
        return  states, actions, rewards
    def sample_env_pre_sampled_from_index(self, start, pre_sample_date_indices, sample_observations,
                                          forward_returns_dates, verbose=False):
        """
        samples environment with pre-sampled dates and paralelized
        :param date_start_index:
        :return:
        """

        dates_indices = pre_sample_date_indices.iloc[start].values.tolist()
        action_dates = forward_returns_dates.index[dates_indices]

        rewards = []
        returns_dates = []
        actions = []
        states = []

        for counter in range(sample_observations):
            action_date = action_dates[counter]
            returns_dates.append(action_date)

            action_date, flat_state, reward, action_portfolio_weights = self._get_sars_by_date(
                action_date=action_date, verbose=False,
                pre_indices=[dates_indices[counter], dates_indices[counter + 1]])

            actions.append(action_portfolio_weights)
            states.append(flat_state)

            rewards.append(reward)

            if action_date > self.max_available_obs_date:
                if verbose:
                    print("Sample reached limit of time series", counter)
                raise

        return states, actions, rewards

    def set_plot_weights(self,weights,benchmark_G):

        self._benchmark_weights=weights
        self._benchmark_G=benchmark_G

    @property
    def benchmark_weights(self):
        return self._benchmark_weights
    @property
    def benchmark_G(self):
        return self._benchmark_G

    def sample_full_env(self):
        """
        samples full environment online training TD-1
        :return:
        """
        states = []
        rewards = []
        actions = []
        for counter, action_date in tqdm(enumerate(self.environment.features.index),desc="sampling full environment"):

            if counter > self.environment.state.in_bars_count:

                next_date, flat_state, reward, action_portfolio_weights = self._get_sars_by_date(
                    action_date=action_date, verbose=False)

                states.append(flat_state)
                rewards.append(reward)
                actions.append(action_portfolio_weights)

        return states, actions,rewards

class LinearAgent(AgentDataBase):

    def __init__(self,*args,**kwargs):
        """



        :param environment:
        :param out_reward_window_td: datetime.timedelta,
        """
        super().__init__(*args,**kwargs)

        self._initialize_linear_parameters()


    def _initialize_linear_parameters(self):
        """
        parameters are for mu and sigma
        (features_rows*features_columns +number_of_assets(weights))*number of asssets
        :return:
        """


        param_dim=self.state_dimension
        self.theta_mu=np.random.rand(self.number_of_assets,param_dim)
        #no modeling correlations if correlation self.theta_sigma=np.random.rand(self.number_of_assets,self.number_of_asset,param_dim)
        self.theta_sigma=np.random.rand(self.number_of_assets,param_dim)

        self.theta_state_baseline=np.random.rand(param_dim)


    def policy(self,flat_state,deterministic=False):
        """
        return action give a linear policy
        :param state:
        :param action_date:
        :return:
        """
        #

        #calculate mu and sigma
        mu=self._mu_linear(flat_state=flat_state)
        sigma=self._sigma_linear(flat_state=flat_state)
        cov = np.zeros((self.number_of_assets, self.number_of_assets))
        np.fill_diagonal(cov, sigma**2)

        try:

            action=np.random.multivariate_normal(
            mu,cov)
        except:
            print("error on sampling")
            raise
        if deterministic:
            return mu
        else:
            return action




    def _state_linear(self,flat_state):
        if isinstance(flat_state,pd.Series):
            flat_state=flat_state.values
        state_value=(self.theta_state_baseline * flat_state).sum()
        return state_value
    def _sigma_linear(self,flat_state):
        if isinstance(flat_state,pd.Series):
            flat_state=flat_state.values
        sigma = np.exp(np.sum(self.theta_sigma * flat_state, axis=1))
        sigma_clip=np.clip(sigma,.05,.2)
        return sigma_clip
    def _mu_linear(self,flat_state):
        if isinstance(flat_state,pd.Series):
            flat_state=flat_state.values

        mu=(self.theta_mu * flat_state).sum(axis=1)
        #clip mu to add up to one , and between .01 and 1 , so no negative values
        c=max(mu)
        mu_clip=np.exp(mu-c) / np.sum(np.exp(mu-c))
        # mu_clip=np.clip(mu,.001,1)
        # mu_clip=mu_clip/np.sum(mu_clip)

        if np.isnan(np.sum(mu_clip)):
            raise

        return mu_clip




    def sample_env(self,observations,verbose=True):
        #starts in 1 becasue comission dependes on initial weights
        start = np.random.choice(range(1,self.latest_posible_index_date))
        start_date =self.environment.features.index[start]
        period_returns = []
        returns_dates=[]
        actions=[]
        states=[]

        for counter,iloc_date in enumerate(range(start, start + observations, 1)):
            if counter==0:
                action_date=start_date

            returns_dates.append(action_date)
            action_date,flat_state,one_period_effective_return,action_portfolio_weights =self._get_sars_by_date(action_date=action_date,verbose=False)

            actions.append(action_portfolio_weights)
            states.append(flat_state.values)


            period_returns.append(one_period_effective_return)

            if action_date > self.max_available_obs_date:
                if verbose:
                     print("Sample reached limit of time series",counter)
                raise

        return states,actions,pd.concat(period_returns, axis=0)


    def ACTOR_CRITIC_FIT(self, alpha=.01, gamma=.99, theta_threshold=.001, max_iterations=10000, plot_gradients=False
                      , record_average_weights=True,  alpha_critic=.01,l_trace=.3,l_trace_critic=.3,use_traces=False):

        theta_diff = 1000
        observations = self.sample_observations
        iters = 0
        n_iters = []
        average_weights = []
        average_reward = []
        theta_norm = []

        pbar = tqdm(total=max_iterations)
        theta_mu_hist_gradients = []
        theta_sigma_hist_gradients = []

        #for plotting
        mu_deterministic=[]
        sigma_deterministic=[]
        V=[]


        while iters < max_iterations:
            n_iters.append(iters)

            # states,actions,period_returns=self.sample_env(observations=observations,verbose=False)
            states, actions, rewards = self.sample_env_pre_sampled(verbose=False)

            average_reward.append(np.mean(rewards))
            new_theta_mu = copy.deepcopy(self.theta_mu)
            new_theta_sigma = copy.deepcopy(self.theta_sigma)

            tmp_mu_gradient = []
            tmp_sigma_gradient = []

            #initialize elegibility traces

            z_theta_critic=np.zeros(self.theta_state_baseline.shape)
            z_theta_mu=np.zeros(self.theta_mu.shape)
            z_theta_sigma=np.zeros(self.theta_sigma.shape)
            I=1
            for t in range(observations):
                action_t = actions[t]
                flat_state_t = states[t]
                if t==observations-1:

                    flat_state_prime_value=0
                else:
                    flat_state_prime=states[t+1]
                    flat_state_prime_value=self._state_linear(flat_state=flat_state_prime)

                delta = rewards[t] + gamma *flat_state_prime_value  - self._state_linear(
                    flat_state=flat_state_t)

                theta_mu_log_gradient = self._theta_mu_log_gradient(action=action_t, flat_state=flat_state_t.values)
                theta_sigma_log_gradient = self._theta_sigma_log_gradient(action=action_t,
                                                                          flat_state=flat_state_t.values)

                if use_traces==True:
                    # traces
                    z_theta_critic = gamma * l_trace_critic * z_theta_critic + self._baseline_linear_gradient(
                        flat_state=flat_state_t)
                    z_theta_mu = gamma * l_trace * z_theta_mu + I * theta_mu_log_gradient
                    z_theta_sigma = gamma * l_trace * z_theta_sigma + I * theta_sigma_log_gradient

                    self.theta_state_baseline = self.theta_state_baseline + alpha_critic * delta * z_theta_critic

                    new_theta_mu = new_theta_mu + alpha * delta *z_theta_mu
                    new_theta_sigma = new_theta_sigma + alpha * delta * z_theta_sigma

                    I=gamma*I
                else:
                    self.theta_state_baseline = self.theta_state_baseline + alpha_critic * delta * self._baseline_linear_gradient(
                        flat_state=flat_state_t)

                    new_theta_mu = new_theta_mu + alpha * delta * (gamma ** t) * theta_mu_log_gradient
                    new_theta_sigma = new_theta_sigma + alpha * delta * (gamma ** t) * theta_sigma_log_gradient

                # save values for plotting
                mu_deterministic.append(self._mu_linear(flat_state=flat_state_t))
                sigma_deterministic.append(self._sigma_linear(flat_state=flat_state_t))
                V.append(self._state_linear(flat_state_t))

                tmp_mu_gradient.append(theta_mu_log_gradient)
                tmp_sigma_gradient.append(theta_sigma_log_gradient)






            theta_mu_hist_gradients.append(np.array(tmp_mu_gradient).mean(axis=1))
            theta_sigma_hist_gradients.append(np.array(tmp_sigma_gradient).mean(axis=1))

            old_full_theta = np.concatenate([self.theta_mu.ravel(), self.theta_sigma.ravel()])
            new_full_theta = np.concatenate([new_theta_mu.ravel(), new_theta_sigma.ravel()])
            # calculate update distance

            theta_diff = np.linalg.norm(new_full_theta - old_full_theta)
            theta_norm.append(theta_diff)
            # print("iteration", iters,theta_diff, end="\r", flush=True)
            pbar.update(1)
            # assign  update_of thetas
            self.theta_mu = copy.deepcopy(new_theta_mu)
            self.theta_sigma = copy.deepcopy(new_theta_sigma)

            iters = iters + 1

            if record_average_weights == True:
                # average_weights.append(self.environment.state.weight_buffer.mean())

                if iters % 200 == 0:

                    ## Create Plot Backtest
                    if not "backtest" in locals():
                        backtest=None
                    backtest=self.backtest_policy(epoch=iters,backtest=backtest)

                    n_cols=len(backtest.columns)
                    for col_counter,col in enumerate(backtest):
                        plt.plot(backtest[col],color="blue",alpha=(col_counter+1)/n_cols)
                    plt.show()
                    #Baccktest plot finishes

                    plt.plot(n_iters, average_reward, label=self.reward_function)
                    plt.plot(n_iters, [self._benchmark_G for i in range(iters)])
                    plt.legend(loc="best")
                    plt.show()

                    mu_chart = np.array(mu_deterministic)
                    sigma_chart = np.array(sigma_deterministic)
                    x_range = [round(i/observations,2) for i in range(mu_chart.shape[0])]
                    cmap = plt.get_cmap('jet')
                    colors = cmap(np.linspace(0, 1.0, mu_chart.shape[1]))
                    for i in range(mu_chart.shape[1]):
                        tmp_mu_plot = mu_chart[:, i]
                        tmp_sigma_plot = sigma_chart[:, i]
                        s_plus = tmp_mu_plot + tmp_sigma_plot
                        s_minus = tmp_mu_plot - tmp_sigma_plot
                        plt.plot(x_range,mu_chart[:, i], label="asset " + str(i), c=colors[i])
                        plt.fill_between(x_range, s_plus, s_minus, color=colors[i], alpha=.2)
                    ws = np.repeat(self._benchmark_weights.reshape(-1, 1), len(x_range), axis=1)
                    for row in range(ws.shape[0]):
                        plt.plot(x_range, ws[row, :], label="benchmark_return" + str(row))
                    plt.legend(loc="best")
                    plt.show()

                    plt.plot(V,label="Value Function")
                    plt.show()

                    if plot_gradients == True:

                        for asset in range(self.number_of_assets):
                            tmp_mu_asset = np.array([i[0, :] for i in theta_mu_hist_gradients])
                            plt.plot(tmp_mu_asset, label=str(asset) + "mu")
                            plt.show()

        return average_weights


    def REINFORCE_fit(self,alpha=.01,gamma=.99,theta_threshold=.001,max_iterations=10000, plot_gradients=False
                             ,record_average_weights=True,add_baseline=False,alpha_baseline=.01):


        theta_diff=1000
        observations=self.sample_observations
        iters=0
        n_iters=[]
        average_weights=[]
        average_reward=[]
        theta_norm=[]

        pbar = tqdm(total=max_iterations)
        theta_mu_hist_gradients=[]
        theta_sigma_hist_gradients=[]

        # for plotting
        mu_deterministic = []
        sigma_deterministic = []

        while iters <max_iterations:
            n_iters.append(iters)

            # states,actions,period_returns=self.sample_env(observations=observations,verbose=False)
            states, actions, rewards = self.sample_env_pre_sampled(verbose=False)

            average_reward.append(np.mean(rewards))
            new_theta_mu=copy.deepcopy(self.theta_mu)
            new_theta_sigma=copy.deepcopy(self.theta_sigma)

            tmp_mu_gradient=[]
            tmp_sigma_gradient=[]
            for t in range(observations):
                action_t=actions[t]
                flat_state_t=states[t]

                gamma_coef=np.array([gamma**(k-t) for k in range(t,observations)])


                G=np.sum(rewards[t:]*gamma_coef)

                if add_baseline==True:
                    delta=G-self._state_linear(flat_state=flat_state_t)
                    self.theta_state_baseline=self.theta_state_baseline+alpha_baseline*delta*self._baseline_linear_gradient(flat_state=flat_state_t)

                else:
                    delta=G

                theta_mu_log_gradient=self._theta_mu_log_gradient(action=action_t,flat_state=flat_state_t.values)
                theta_sigma_log_gradient=self._theta_sigma_log_gradient(action=action_t,flat_state=flat_state_t.values)

                tmp_mu_gradient.append(theta_mu_log_gradient)
                tmp_sigma_gradient.append(theta_sigma_log_gradient)


                new_theta_mu=new_theta_mu+alpha*delta*(gamma**t)*theta_mu_log_gradient
                new_theta_sigma=new_theta_sigma+alpha*delta*(gamma**t)*theta_sigma_log_gradient

                # save values for plotting
                mu_deterministic.append(self._mu_linear(flat_state=flat_state_t))
                sigma_deterministic.append(self._sigma_linear(flat_state=flat_state_t))

            theta_mu_hist_gradients.append(np.array(tmp_mu_gradient).mean(axis=1))
            theta_sigma_hist_gradients.append(np.array(tmp_sigma_gradient).mean(axis=1))

            old_full_theta=np.concatenate([self.theta_mu.ravel(),self.theta_sigma.ravel()])
            new_full_theta=np.concatenate([new_theta_mu.ravel(),new_theta_sigma.ravel()])
                #calculate update distance

            theta_diff=np.linalg.norm(new_full_theta-old_full_theta)
            theta_norm.append(theta_diff)
            # print("iteration", iters,theta_diff, end="\r", flush=True)
            pbar.update(1)
            #assign  update_of thetas
            self.theta_mu=copy.deepcopy(new_theta_mu)
            self.theta_sigma=copy.deepcopy(new_theta_sigma)

            iters=iters+1

            if record_average_weights==True:

                #Todo: implement in tensorboard
                if iters%200==0:

                    plt.plot(n_iters, average_reward, label=self.reward_function)
                    plt.plot(n_iters, [self._benchmark_G for i in range(iters)])
                    plt.legend(loc="best")
                    plt.show()

                    mu_chart = np.array(mu_deterministic)
                    sigma_chart = np.array(sigma_deterministic)
                    x_range = [round(i / observations, 2) for i in range(mu_chart.shape[0])]
                    cmap = plt.get_cmap('jet')
                    colors = cmap(np.linspace(0, 1.0, mu_chart.shape[1]))
                    for i in range(mu_chart.shape[1]):
                        tmp_mu_plot = mu_chart[:, i]
                        tmp_sigma_plot = sigma_chart[:, i]
                        s_plus = tmp_mu_plot + tmp_sigma_plot
                        s_minus = tmp_mu_plot - tmp_sigma_plot
                        plt.plot(mu_chart[:, i], label="asset " + str(i), c=colors[i])
                        plt.fill_between(x_range, s_plus, s_minus, color=colors[i], alpha=.2)
                    ws = np.repeat(self._benchmark_weights.reshape(-1, 1), len(x_range), axis=1)
                    for row in range(ws.shape[0]):
                        plt.plot(x_range, ws[row, :], label="benchmark_return" + str(row))
                    plt.legend(loc="best")
                    plt.show()

                    if plot_gradients==True:

                        for asset in range(self.number_of_assets):
                            tmp_mu_asset=np.array([i[0,:] for i in theta_mu_hist_gradients])
                            plt.plot(tmp_mu_asset,label=str(asset)+"mu")
                            plt.show()

        return average_weights


    def _theta_mu_log_gradient(self,action,flat_state):
        """

        :param action: pd.DataFrame
        :param flat_state: pd.DataFrame
        :return:
        """
        sigma=self._sigma_linear(flat_state=flat_state)
        mu=self._mu_linear(flat_state=flat_state)
        denominator=1/sigma**2
        log_gradient=(denominator*(action-mu)).reshape(-1,1)*(flat_state)

        return  log_gradient

    def _theta_sigma_log_gradient(self,action,flat_state):
        """

        :param action:
        :param flat_state:
        :return:
        """
        sigma = self._sigma_linear(flat_state=flat_state)
        mu = self._mu_linear(flat_state=flat_state)
        log_gradient=(((action-mu)/sigma)**2 -1).reshape(-1,1)*flat_state
        return  log_gradient

    def _baseline_linear_gradient(self,flat_state):
        return flat_state


import copy


class PolicyEstimator(torch.nn.Module):
    def __init__(self, state_dimension,number_of_assets):
        super(PolicyEstimator, self).__init__()
        self.state_dimension=state_dimension
        self.number_of_assets=number_of_assets

        #Todo: Main Core of the model Architecture
        self.mus = torch.nn.Linear(self.state_dimension, self.number_of_assets)
        self.log_sigmas = torch.nn.Linear(self.state_dimension, self.number_of_assets)

    def forward(self,x):
        """

        :param x:
        :return:
        """
        sigmas=torch.exp(self.log_sigmas(x))
        clip_sigmas=torch.clamp(sigmas,.01,.2)
        mus_clip=torch.nn.Softmax()(self.mus(x))
        return mus_clip,clip_sigmas


class ActorEstimator(torch.nn.Module):
    def __init__(self, state_dimension, number_of_assets):
        super(ActorEstimator, self).__init__()
        self.state_dimension = state_dimension
        self.number_of_assets = number_of_assets
        # Todo: Main Core of the model Architecture
        self.state_value_f=torch.nn.Linear(self.state_dimension,1)

    def forward(self,x):
        """

        :param x:
        :return:
        """
        state=self.state_value_f(x)
        return state


class DeepAgentPytorch(AgentDataBase):



    def __init__(self,*args,**kwargs):
        super().__init__(*args, **kwargs)


        self.build_model()
        self.train_average_weights = []

    def CustomLossGaussian(self,state, action, reward):
        # Obtain mu and sigma from actor network
        nn_mu, nn_sigma = self.actor_model(state)


        # Obtain pdf of Gaussian distribution
        pdf_value = torch.exp(-0.5 * ((action - nn_mu) / (nn_sigma)) ** 2) * \
                    1 / (nn_sigma * np.sqrt(2 * np.pi))

        # Compute log probability
        log_probability = torch.log(pdf_value + 1e-5)

        # Compute weighted loss
        loss_actor = reward * log_probability
        #reduce mean have local minim
        J=torch.mean(torch.sum(loss_actor,axis=1))
        return -J

    def CustomLossCritic(self,state,advantages):
        """
        custom loss for Critic
        :param state:
        :param advantages:
        :return:
        """
        state_values=self.critic_model(state)
        loss= F.smooth_l1_loss(state_values,advantages)
        return loss

    def policy(self,flat_state):
        """
        Policy needs to use a prediction from the deep model
        :param flat_state:
        :return:
        """
        if isinstance(flat_state,pd.Series):
            input=torch.FloatTensor(np.array([flat_state.values]).reshape(-1))
        else:
            input=torch.FloatTensor(flat_state)

        mu, sigmas=self.actor_model(input)
        mu=mu.detach().numpy()
        sigmas=sigmas.detach().numpy()

        cov = np.zeros((self.number_of_assets, self.number_of_assets))
        np.fill_diagonal(cov, sigmas ** 2)

        try:

            action = np.random.multivariate_normal(
                mu, cov)


        except:
            print("error on sampling")
            raise

        return action


    def build_model(self):
        # TODO: Normalization needs to done pre-batch. Here is been done with one batch it seems to me.

        self.actor_model=PolicyEstimator(state_dimension=self.state_dimension,number_of_assets=self.number_of_assets)
        self.critic_model=ActorEstimator(state_dimension=self.state_dimension,number_of_assets=self.number_of_assets)

    def ACTOR_CRITIC_fit(self,gamma=.99, max_iterations=10000,record_average_weights=True):

        observations = self.sample_observations
        iters = 0
        n_iters = []
        average_weights = []
        average_reward = []
        total_reward = []
        theta_norm = []
        losses = []
        pbar = tqdm(total=max_iterations)

        optimizer = torch.optim.Adam(self.actor_model.parameters(),
                                     lr=0.01)

        historical_grads = []

        #for plotting
        mus_deterministic=[]
        sigma_deterministc=[]
        V=[]
        while iters < max_iterations:
            n_iters.append(iters)


            states, actions, rewards = self.sample_env_pre_sampled(verbose=False)
            states = np.array([s.values for s in states]).reshape(self.sample_observations, -1)
            average_reward.append(np.mean(rewards))
            actions = np.array(actions)
            advantages=[]
            states_tensor = torch.FloatTensor(states)
            actions_tensor = torch.FloatTensor(actions)


            for t in range(observations):

                flat_state_t = states_tensor[t]
                if t == observations - 1:

                    flat_state_prime_value=0
                else:
                    flat_state_prime = states_tensor[t + 1]
                    flat_state_prime_value=self.critic_model(flat_state_prime).detach().numpy()
                #delta is also advantage

                delta = rewards[t] + gamma * flat_state_prime_value - self.critic_model(flat_state_t).detach().numpy()

                advantages.append(delta[0])
                tmp_mu,temp_s=self.actor_model(flat_state_t)
                mus_deterministic.append(tmp_mu.detach().numpy())
                sigma_deterministc.append(temp_s.detach().numpy())
                V.append(self.critic_model(flat_state_t).detach().numpy())

            As=np.array(advantages)
            As=As.reshape(-1,1)
            As_tensor = torch.FloatTensor(As)


            optimizer.zero_grad()

            loss_actor = self.CustomLossGaussian(states_tensor, actions_tensor, As_tensor)
            loss_critic= self.CustomLossCritic(state=states_tensor,advantages=As_tensor)

            # sum up all the values of policy_losses and value_losses
            loss_value = loss_actor+ loss_critic

            # calculate gradients
            loss_value.backward()

            # apply gradients
            optimizer.step()

            pbar.update(1)
            iters = iters + 1
            # historical_grads.append(loss_value.grad.numpy())

            pbar.set_description("loss " + str(loss_value))
            losses.append(float(loss_value))

            if record_average_weights == True:

                if iters % 200 == 0:


                    mu_chart = np.array(mus_deterministic)
                    sigma_chart=np.array(sigma_deterministc)
                    x_range=range(mu_chart.shape[0])
                    cmap = plt.get_cmap('jet')
                    colors = cmap(np.linspace(0, 1.0, mu_chart.shape[1]))
                    for i in range(mu_chart.shape[1]):
                        tmp_mu_plot=mu_chart[:,i]
                        tmp_sigma_plot=sigma_chart[:,i]
                        s_plus=tmp_mu_plot+tmp_sigma_plot
                        s_minus = tmp_mu_plot -tmp_sigma_plot
                        plt.plot(mu_chart[:,i],label="asset "+ str(i),c=colors[i])
                        plt.fill_between(x_range,s_plus,s_minus,color=colors[i],alpha=.2)
                    ws = np.repeat(self._benchmark_weights.reshape(-1, 1), len(x_range), axis=1)
                    for row in range(ws.shape[0]):
                        plt.plot(x_range, ws[row, :], label="benchmark_return" + str(row))
                    plt.legend(loc="best")
                    plt.show()


                    plt.plot(n_iters, average_reward, label=self.reward_function)
                    plt.plot(n_iters, [self._benchmark_G for i in range(iters)])
                    plt.legend(loc="best")
                    plt.show()

                    plt.plot(V, label="Value Function")
                    plt.show()


    def REINFORCE_fit(self,  gamma=.99, max_iterations=10000
                      , record_average_weights=True):


        observations = self.sample_observations
        iters = 0
        n_iters = []
        average_weights = []
        average_reward = []
        total_reward=[]
        theta_norm = []
        losses=[]
        pbar = tqdm(total=max_iterations)

        optimizer = torch.optim.Adam(self.actor_model.parameters(),
                               lr=0.01)

        historical_grads = []
        while iters < max_iterations:
            n_iters.append(iters)

            # states,actions,period_returns=self.sample_env(observations=observations,verbose=False)
            states, actions, rewards = self.sample_env_pre_sampled(verbose=False)
            states=np.array([s.values for s in states]).reshape(self.sample_observations,-1)
            average_reward.append(np.mean(rewards))
            # total_reward.extend(rewards)
            actions=np.array(actions)
            Gs=[]

            for t in range(observations):

                gamma_coef = np.array([gamma ** (k - t) for k in range(t, observations)])

                G = np.sum(rewards[t:] * gamma_coef)
                Gs.append(G)
            Gs=np.array(Gs)
            Gs=Gs.reshape(-1,1)
            Gs=Gs

            Gs_tensor=torch.FloatTensor(Gs)
            states_tensor=torch.FloatTensor(states)
            actions_tensor=torch.FloatTensor(actions)

            optimizer.zero_grad()

            loss_value = self.CustomLossGaussian(states_tensor, actions_tensor, Gs_tensor)
            #calculate gradients
            loss_value.backward()
            #apply gradients
            optimizer.step()

            pbar.update(1)
            iters = iters + 1
            # historical_grads.append(loss_value.grad.numpy())

            pbar.set_description("loss "+str(loss_value))
            losses.append(float(loss_value))
            if record_average_weights == True:
                average_weights.append(self.environment.state.weight_buffer.mean())
                # Todo: implement in tensorboard
                if iters % 200 == 0:

                    weights = pd.concat(average_weights, axis=1).T
                    ax = weights.plot()
                    ws = np.repeat(self._benchmark_weights.reshape(-1, 1), len(average_weights), axis=1)
                    for row in range(ws.shape[0]):
                        ax.plot(n_iters, ws[row, :], label="benchmark_return" + str(row))
                    plt.legend(loc="best")
                    plt.show()

                    plt.plot(n_iters, average_reward, label=self.reward_function)
                    plt.plot(n_iters, [self._benchmark_G for i in range(iters)])
                    plt.legend(loc="best")
                    plt.show()

                    # plt.plot(total_reward,label="total_reward")
                    # plt.show()


                    # plt.plot(losses,legend="losses")

                    # grads_to_plot_1=np.array([i[:,0] for i in historical_grads])
                    # grads_to_plot_2=np.array([i[:,1] for i in historical_grads])
                    # plt.plot(grads_to_plot_1)
                    # plt.plot(grads_to_plot_2)
                    # plt.show()

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)