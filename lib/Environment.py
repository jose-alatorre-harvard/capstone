import gym
import pandas as pd
import numpy as np
import os
import datetime
from tqdm import tqdm
from lib.Benchmarks import SimulatedAsset
from lib.DataHandling import DailyDataFrame2Features
import matplotlib.pyplot as plt
import copy
import warnings
from joblib import Parallel, delayed


class State:

    def __init__(self, features,forward_returns,forward_returns_dates, objective_parameters):
        """

          :param features:
          :param forward_returns:
          :param forward_returns_dates:
          :param objective_parameters:
        """

        self.features = features
        self.forward_returns=forward_returns
        self.forward_returns_dates=forward_returns_dates
        self._set_helper_functions()
        self._set_objective_function_parameters(objective_parameters)

        self._initialize_weights_buffer()

    def flatten_state(self,state_features, weights_on_date):
        """
        flatten states by adding weights to features

        :return:
        """
        flat_state=state_features.copy()
        for index in weights_on_date.index:
            flat_state[index]=weights_on_date.loc[index]

        return flat_state

    def _set_helper_functions(self):
        """
        Creates following properties
        assets_names: (list)
        log_close_returns: (pd.DataFrame)
        :return:
        """

        self.number_of_assets=len(self.forward_returns.columns)
        self.state_dimension=self.features.shape[1] +self.number_of_assets



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

    def step(self, action, action_date,pre_indices=None):
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


        weight_difference = self.weight_buffer.iloc[action_date_index - 1:action_date_index + 1]
        # obtain the difference from the previous allocation, diff is done t_1 - t
        weight_difference = abs(weight_difference.diff().dropna())

        # calculate rebalance commission
        commision_percent_cost = -weight_difference.sum(axis = 1) * self.percent_commission

        # get period_ahead_returns
        t_plus_one_returns = self.forward_returns.iloc[action_date_index]
        one_period_mtm_reward = (t_plus_one_returns * action).sum()

        #mtm - commissions
        one_period_effective_return = one_period_mtm_reward - commision_percent_cost

        if len(one_period_effective_return) == 0:
            raise

        done= False
        extra_info={"action_date":action_date,
            "forward_returns":t_plus_one_returns,
                    "previous_weights":self.weight_buffer.iloc[action_date_index - 1]}
        return next_observation_date,one_period_effective_return,done,extra_info

    def encode(self, date):
        """
        convert current state to tensor

        """
        pass

    def get_state_on_date(self, target_date,pre_indices=None):
        """
            returns the state on a target date
           :param target_date:
           :return: in_window_features, weights_on_date
        """
        #TODO: what happens for  different features for example ("Non Time Series Returns")?
        assert target_date >= self.features.index[0]
        if pre_indices is None:
            date_index = self.features.index.searchsorted(target_date)
        else:
            date_index=pre_indices[0]
        state_features =self.features.iloc[date_index]
        weights_on_date = self.weight_buffer.iloc[date_index]

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
        if state_type =="in_window_out_window":
            # Will be good if meta parameters does not need to be passed even to the environment possible?
            self.state = State(features=self.features,

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

    def step(self, action_portfolio_weights, action_date,pre_indices=None):
        """

        :param action_portfolio_weights:
        :param action_date:
        :return:
        """

        action = action_portfolio_weights
        observation,reward,done,extra_info= self.state.step(action, action_date,pre_indices)
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

    def __init__(self,environment,out_reward_window_td,reward_function,sample_observations=32):
        """



        :param environment:
        :param out_reward_window_td: datetime.timedelta,
        """
        self.environment = environment
        self.out_reward_window_td=out_reward_window_td
        self.sample_observations = sample_observations
        self.reward_function=reward_function
        self._initialize_helper_properties()
        self._initialize_linear_parameters()


        self._set_latest_posible_date()



    def _initialize_helper_properties(self):

        self.number_of_assets=self.environment.number_of_assets
        self.state_dimension=self.environment.state.state_dimension





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
        self.max_available_obs_date=frd[column_name].index.max()

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


    def _policy_linear(self,flat_state):
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

        return action

    def _sigma_linear(self,flat_state):
        sigma = np.exp(np.sum(self.theta_sigma * flat_state.values, axis=1))
        sigma_clip=np.clip(sigma,.05,.2)
        return sigma_clip
    def _mu_linear(self,flat_state):
        mu=(self.theta_mu * flat_state.values).sum(axis=1)
        #clip mu to add up to one , and between .01 and 1 , so no negative values

        mu_clip=np.exp(mu) / np.sum(np.exp(mu))
        # mu_clip=np.clip(mu,.001,1)
        # mu_clip=mu_clip/np.sum(mu_clip)

        if np.isnan(np.sum(mu_clip)):
            raise

        return mu_clip

    def get_best_action(self,flat_state):
        """
        returns best action given state (portfolio weights
        :param state:
        :param action_date:
        :return:
        """

        action=self._policy_linear(flat_state=flat_state)


        return action



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

        next_action_date, one_period_effective_return, done, info = self.environment.step(
            action_portfolio_weights=action_portfolio_weights,
            action_date=action_date,pre_indices=pre_indices)

        if verbose:

            print(info)

        return next_action_date,  flat_state ,one_period_effective_return, action_portfolio_weights


    def sample_env_pre_sampled(self,verbose=False):
        """
        samples environment with pre-sampled dates and paralelized
        :param date_start_index:
        :return:
        """

        # starts in 1 becasue comission dependes on initial weights
        start = np.random.choice(range(1, self.latest_posible_index_date))

        dates_indices=self.pre_sample_date_indices.iloc[start].values.tolist()
        action_dates=self.environment.forward_returns_dates.index[dates_indices]

        period_returns = []
        returns_dates = []
        actions = []
        states = []

        for counter in range(self.sample_observations):
            action_date=action_dates[counter]
            returns_dates.append(action_date)

            action_date, flat_state, one_period_effective_return, action_portfolio_weights = self._get_sars_by_date(
                action_date=action_date, verbose=False,pre_indices=[dates_indices[counter],dates_indices[counter+1]])

            actions.append(action_portfolio_weights)
            states.append(flat_state)

            period_returns.append(one_period_effective_return)

            if action_date > self.max_available_obs_date:
                if verbose:
                    print("Sample reached limit of time series", counter)
                raise

        return states, actions, pd.concat(period_returns, axis=0)



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
            states.append(flat_state)


            period_returns.append(one_period_effective_return)

            if action_date > self.max_available_obs_date:
                if verbose:
                     print("Sample reached limit of time series",counter)
                raise

        return states,actions,pd.concat(period_returns, axis=0)
    def REINFORCE_linear_fit(self,alpha=.1,gamma=.99,theta_threshold=.001,max_iterations=20000
                             ,record_average_weights=True):

        theta_diff=1000
        observations=self.sample_observations
        iters=0
        n_iters=[]
        average_weights=[]
        G_0s=[]
        theta_norm=[]

        pbar = tqdm(total=max_iterations)
        while iters <max_iterations:
            n_iters.append(iters)

            # states,actions,period_returns=self.sample_env(observations=observations,verbose=False)
            states, actions, period_returns = self.sample_env_pre_sampled(verbose=False)

            rewards=self.returns_to_reward_factory(period_returns=period_returns)
            new_theta_mu=copy.deepcopy(self.theta_mu)
            new_theta_sigma=copy.deepcopy(self.theta_sigma)
            for t in range(observations):
                action_t=actions[t]
                flat_state_t=states[t]

                gamma_coef=np.array([gamma**(k-t) for k in range(t,observations)])


                G=np.sum(rewards[t:]*gamma_coef)

                if t==0:
                    G_0s.append(G)

                new_theta_mu=new_theta_mu+alpha*G*(gamma**t)*self._theta_mu_log_gradient(action=action_t,flat_state=flat_state_t)
                new_theta_sigma=new_theta_sigma+alpha*G*(gamma**t)*self._theta_sigma_log_gradient(action=action_t,flat_state=flat_state_t)

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
                average_weights.append(self.environment.state.weight_buffer.mean())
                #Todo: implement in tensorboard
                if iters%2000==0:
                    weights=pd.concat(average_weights, axis=1).T
                    ax=weights.plot()
                    ws=np.repeat(self._benchmark_weights.reshape(-1,1),iters,axis=1)
                    for row in range(ws.shape[0]):
                        ax.plot(n_iters,ws[row,:],label="benchmark_return"+str(row))
                    plt.legend(loc="best")
                    plt.show()

                    plt.plot(n_iters,G_0s,label=self.reward_function)
                    plt.plot(n_iters,[self._benchmark_G for i in range(iters)])
                    plt.legend(loc="best")
                    plt.show()

                    plt.plot(n_iters,theta_norm,label="norm improvement")
                    plt.legend(loc="best")
                    plt.show()

                    alpha=alpha/2
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
        log_gradient=(denominator*(action-mu)).reshape(-1,1)*(flat_state.values)

        return  log_gradient

    def _theta_sigma_log_gradient(self,action,flat_state):
        """

        :param action:
        :param flat_state:
        :return:
        """
        sigma = self._sigma_linear(flat_state=flat_state)
        mu = self._mu_linear(flat_state=flat_state)
        log_gradient=(((action-mu)/sigma)**2 -1).reshape(-1,1)*flat_state.values
        return  log_gradient

    def returns_to_reward_factory(self, period_returns):
        """
        launch reward types Needs to be implemented
        :param reward:
        :return:
        """
        reward_function = self.reward_function
        if reward_function == "cum_return":
            return self._reward_cum_return(period_returns)
        elif reward_function == "max_sharpe":
            return self._reward_max_sharpe(period_returns)
        elif reward_function == "min_vol":
            return self._reward_to_min_vol(period_returns)

    def _reward_to_min_vol(self,period_returns):
        """
        minimum volatility portfolio
        :param period_returns:
        :return:
        """
        rewards = np.zeros(period_returns.shape[0])
        vol = period_returns.std() * np.sqrt(252 / 7)
        rewards[-1] = -vol

        return rewards

    def _reward_max_sharpe(self, period_returns):
        """
        calculates sharpe ratio for the returns
        :param period_returns:
        :return:
        """

        warnings.warn('Using  anualization factor as daily')
        rewards=np.zeros(period_returns.shape[0])

        mean_return=period_returns.mean()*(252/7)
        vol=period_returns.std()*np.sqrt(252/7)
        sharpe=mean_return/(vol)
        rewards[-1]=sharpe



        return rewards



    def _reward_cum_return(self, period_returns):

        return period_returns.values

    def set_plot_weights(self,weights,benchmark_G):

        self._benchmark_weights=weights
        self._benchmark_G=benchmark_G
