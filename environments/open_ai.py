import gym
import pandas as pd
import numpy as np
import torch

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
            function_rewards=self._reward_cum_return(portfolio_returns)

        elif reward_function == "max_sharpe":
            function_rewards = self._reward_max_sharpe(portfolio_returns)
        elif reward_function == "min_vol":
            function_rewards= self._reward_to_min_vol(portfolio_returns)
        elif reward_function == "min_realized_variance":
            function_rewards=self._min_realized_variance(portfolio_returns)

        return function_rewards



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

    def _min_realized_variance(self,portfolio_returns):
        """

        :param portfolio_returns:
        :return:
        """
        return -portfolio_returns.iloc[-1]**2
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
            state = self.get_flat_state_by_iloc(index_location=index_location + 1)
            states.append(state.values)
        return states

    def update_weights_by_iloc(self,index_location,sample_size,new_weights):
        """

        :param index_location:
        :param sample_size:
        :return:
        """
        self.weight_buffer.iloc[index_location:index_location+sample_size]=new_weights

    def step(self, action_date_index, action, reward_function):
        """

        :param action_date_index:
        :param action:
        :return:
        """
        MIN_LEVERAGE=.9
        MAX_LEVERAGE=1.1

        if action_date_index == self.forward_returns_dates.shape[0]-1:
            #pass done if episode is end of time series
            done =True
            next_observation_date_index=None
        else:
            done = False

            next_observation_date = self.forward_returns_dates.iloc[action_date_index].values[0]
            next_observation_date_index = self.weight_buffer.index.searchsorted(next_observation_date)

            self.weight_buffer.iloc[action_date_index:next_observation_date_index, :] = action

        reward = self.reward_factory.get_reward(weights_bufffer=self.weight_buffer,
                                                forward_returns=self.forward_returns,
                                                action_date_index=action_date_index,
                                                reward_function=reward_function)

        # reward=reward-abs(1-action.sum())*abs(reward)/2

        extra_info = {}
        obs=self.get_flat_state_by_iloc(index_location=action_date_index)

        return next_observation_date_index, reward, done,obs, extra_info

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
    """
    Trading Environment
    https://github.com/openai/gym/blob/master/gym/core.py
    """

    def __init__(self, features, forward_returns, forward_returns_dates, objective_parameters,
                 meta_parameters,observations_per_episode=32):
        """
          features and forward returns should be aligned by the time axis. The setup should resemble a supervised learning

          :param features: pandas.DataFrame, historical features
          :param forward_returns: pandas.DataFrame, assets forward returns
          :param objective_parameters:
          :param meta_parameters:
        """
        super().__init__()

        assert features.index.equals(forward_returns.index)

        self.features = features
        self.forward_returns = forward_returns
        self.forward_returns_dates = forward_returns_dates
        # create helper variables
        self._set_environment_helpers()
        self.objective_parameters = objective_parameters
        self.observations_per_episode=observations_per_episode

        self._set_state(meta_parameters=meta_parameters, objective_parameters=objective_parameters)

        # action space is the portfolio weights at any time in our example it is bounded by [0,1]
        self.action_space = gym.spaces.Box(low=0, high=1, shape=(self.number_of_assets,))
        # features to be scaled normal scaler will bound them in -4,4
        self.observation_space = gym.spaces.Box(low=-4, high=4, shape=(self.number_of_features,))

        self.reset()

    def _set_environment_helpers(self):
        """
        creates helper variables for the environment
        """
        self.number_of_assets = len(self.forward_returns.columns)
        self.number_of_features=len(self.features.columns)
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
    def reset(self):
        """Resets the environment to an initial state and returns an initial
        observation.
        Note that this function should not reset the environment's random
        number generator(s); random variables in the environment's state should
        be sampled independently between multiple calls to `reset()`. In other
        words, each call of `reset()` should yield an environment suitable for
        a new episode, independent of previous episodes.

        The reset should start in a random point in time. It is important to notice that even when
        asset trading is a continuous problem  we are clipping the future environment information as to a prediction
        range that fits the problem. For example if we are using a history of intraday data of 3 hours, there is no
        reason to include rewards for the weeks, months.

        Returns:
            observation (object): the initial observation.
        """

        #1 on each reset generate a random number that points to a random start date in the iloc of the index

        self.current_action_index=np.random.choice(range(self.state.in_bars_count + 1, self.features.shape[0]-2))
        self.step_count=0


        #Todo: Reset weight buffer
        initial_observation=self.state.get_flat_state_by_iloc(index_location=self.current_action_index)

        return initial_observation


    def step(self,action):
        """Run one timestep of the environment's dynamics. When end of
                episode is reached, you are responsible for calling `reset()`
                to reset this environment's state.
                Accepts an action and returns a tuple (observation, reward, done, info).
                Args:
                    action (object): an action provided by the agent
                Returns:
                    observation (object): agent's observation of the current environment
                    reward (float) : amount of reward returned after previous action
                    done (bool): whether the episode has ended, in which case further step() calls will return undefined results
                    info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """




        next_observation_date_index, reward, done,obs, extra_info=self.state.step(action_date_index=self.current_action_index,action=action,
                        reward_function=self.objective_parameters["reward_function"])



        # do maitenance function to the environment
        self.current_step =next_observation_date_index
        self.step_count=self.step_count+1
        if self.step_count == self.observations_per_episode:
            done =True
            self.reset()



        return obs, reward, done, {}


    def render(self, mode='human'):
        """Renders the environment.
        The set of supported modes varies per environment. (And some
        environments do not support rendering at all.) By convention,
        if mode is:
        - human: render to the current display or terminal and
          return nothing. Usually for human consumption.
        - rgb_array: Return an numpy.ndarray with shape (x, y, 3),
          representing RGB values for an x-by-y pixel image, suitable
          for turning into a video.
        - ansi: Return a string (str) or StringIO.StringIO containing a
          terminal-style text representation. The text can include newlines
          and ANSI escape sequences (e.g. for colors).
        Note:
            Make sure that your class's metadata 'render.modes' key includes
              the list of supported modes. It's recommended to call super()
              in implementations to use the functionality of this method.
        Args:
            mode (str): the mode to render with
        Example:
        class MyEnv(Env):
            metadata = {'render.modes': ['human', 'rgb_array']}
            def render(self, mode='human'):
                if mode == 'rgb_array':
                    return np.array(...) # return RGB frame suitable for video
                elif mode == 'human':
                    ... # pop up a window and render
                else:
                    super(MyEnv, self).render(mode=mode) # just raise an exception
        """


        print(f'Step :{self.current_step}')




