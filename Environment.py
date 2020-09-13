import gym
import pandas as pd
import numpy as np
import os
import datetime
from tqdm import tqdm



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

        assert target_date >= self._features.index[self.in_bars_count]

        date_index = self._features.index.searchsorted(target_date)
        state_features = self._features.iloc[date_index - self.in_bars_count + 1:date_index + 1]
        weights_on_date = self.weight_buffer.iloc[date_index]

        return state_features, weights_on_date


class DeepTradingEnvironment(gym.Env):
    metadata = {'render.modes': ['human']}

    @classmethod
    def build_from_simulated_assets(cls,number_of_assets,simulation_method="GBM"):
        """
        builds environment from simulated assets
        :param number_of_assets:
        :param simulation_method:
        :return:
        """

    @staticmethod
    def _buid_close_returns(assets_prices, out_reward_window):
        """
         builds close-to-close returns for a specif
         :param self:
         :param assets_prices:(DataFrame)
         :param out_reward_window:(datetime.timedelta)
         :return:
         """

        PERSISTED_DATA_DIRECTORY = "temp_persisted_data"
        # Todo: Hash csv file
        if not os.path.exists(PERSISTED_DATA_DIRECTORY + "/log_close_returns"):
            try:
                log_close_returns = assets_prices.copy() * np.nan
                cross_over_day=log_close_returns[log_close_returns.columns[0]].fillna(0)
                next_return_time_stamp = log_close_returns[log_close_returns.columns[0]].fillna(0)
                for counter, date in tqdm(enumerate(assets_prices.index)):
                    next_date = date + out_reward_window

                    next_date_index = assets_prices.index.searchsorted(next_date)
                    log_close_returns.iloc[counter] = np.log(
                        assets_prices.iloc[next_date_index] / assets_prices.iloc[counter])

                    next_return_time_stamp.iloc[counter]=assets_prices.index[next_date_index]
                    if assets_prices.index[next_date_index].date() != assets_prices.index[counter].date():
                         # cross over date
                         cross_over_day.iloc[counter]=1


            except:
                log_close_returns.iloc[counter] = np.nan


            log_close_returns.to_parquet(PERSISTED_DATA_DIRECTORY + "/log_close_returns")
            cross_over_day=pd.DataFrame(cross_over_day)
            cross_over_day.to_parquet(PERSISTED_DATA_DIRECTORY + "/cross_over_day")
            next_return_time_stamp=pd.DataFrame(next_return_time_stamp)
            next_return_time_stamp.to_parquet(PERSISTED_DATA_DIRECTORY + "/next_return_time_stamp")
        else:

            log_close_returns = pd.read_parquet(PERSISTED_DATA_DIRECTORY + "/log_close_returns")
            cross_over_day = pd.read_parquet(PERSISTED_DATA_DIRECTORY + "/cross_over_day")
            next_return_time_stamp = pd.read_parquet(PERSISTED_DATA_DIRECTORY + "/next_return_time_stamp")

        return {"input_features":assets_prices,
            "log_close_returns":log_close_returns,"cross_over_day":cross_over_day,"next_return_time_stamp":next_return_time_stamp}

    @classmethod
    def from_dirs_and_transform(cls, meta_parameters, objective_parameters, data_dir="data_env", **kwargs):
        """
        Do transformations that shouldnt be part of the class

        Also using the meta parameters


        """

        # optimally this should be only features
        assets_prices = {file: pd.read_parquet(data_dir + "/" + file)["close"] for file in os.listdir(data_dir)}

        assets_prices = pd.DataFrame(assets_prices)

        #resample
        assets_prices=assets_prices.resample("5min").first()
        assets_prices=assets_prices.dropna()
        input_data=cls._buid_close_returns(assets_prices=assets_prices, out_reward_window=meta_parameters["out_reward_window"])
        # transform features



        return DeepTradingEnvironment(input_data=input_data,
                                      assets_prices=assets_prices, meta_parameters=meta_parameters,
                                      objective_parameters=objective_parameters, **kwargs)



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

        self._set_state(meta_parameters=meta_parameters)


        # action space is the portfolio weights at any time in our example it is bounded by [0,1]
        self.action_space = gym.spaces.Box(low=0, high=1, shape=(self.number_of_assets,))
        # features to be scaled normal scaler will bound them in -4,4
        self.observation_space = gym.spaces.Box(low=-4, high=4, shape=(self.number_of_features,))




    def _set_state(self,meta_parameters):
        # logic to create state
        state_type=meta_parameters["state_type"]
        if state_type =="in_window_out_window":
            # Will be good if meta parameters does not need to be passed even to the environment possible?
            self._state = State(features=self.features,
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
        observation,reward,done,extra_info= self._state.step(action, action_date)
        # obs = self._state.encode()
        obs=observation
        info = {"action": action,
                "date": action_date}

        return obs, reward, done, info

    def render(self, mode='human', close=False):
        pass

    def close(self):
        pass

    def generate_episodes(self,observations=100):

        #generate random start
        start=np.random.choice(range(self.assets_prices.shape[0]))

        rewards=[]
        #Todo: Note that sum of rewards assume that we take a bet on each step ok for training?
        for iloc_date in range(start,start+observations,1):

            action_portfolio_weights=np.random.rand(2)
            obs, reward, done, info=self.step(action_portfolio_weights=action_portfolio_weights,action_date=self.assets_prices.index[iloc_date])

            rewards.append(reward)

class TraderTrainer:

    def __init__(self,environment,agent):
        pass

# parameters related to the transformation of data, this parameters govern an step before the algorithm
meta_parameters = {"in_bars_count": 30,
                   "out_reward_window": datetime.timedelta(minutes=10),
                   "state_type":"in_window_out_window"}

# parameters that are related to the objective/reward function construction
objective_parameters = {"percent_commission": .001}
print("===Meta Parameters===")
print(meta_parameters)
print("===Objective Parameters===")
print(objective_parameters)

env=DeepTradingEnvironment.from_dirs_and_transform(meta_parameters=meta_parameters,objective_parameters=objective_parameters)

action=np.random.rand(2)
env.generate_episodes()