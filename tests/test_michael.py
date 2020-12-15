
from environments.e_greedy import DeepTradingEnvironment, LinearAgent,DeepAgentPytorch
import datetime
import numpy as np
import pandas as pd

out_reward_window=datetime.timedelta(days=1)
# parameters related to the transformation of data, this parameters govern an step before the algorithm
meta_parameters = {"in_bars_count": 30,
                   "out_reward_window":out_reward_window ,
                   "state_type":"in_window_out_window",
                   "include_previous_weights":False}

# parameters that are related to the objective/reward function construction
objective_parameters = {"percent_commission": .001}
features=pd.read_parquet("/home/jose/Downloads/features_df")
forward_returns=pd.read_parquet("/home/jose/Downloads/forward_returns_df")
forward_returns_dates=pd.read_parquet("/home/jose/Downloads/forward_returns_dates")
env=DeepTradingEnvironment(features, forward_returns, forward_returns_dates, objective_parameters,
                 meta_parameters)

linear_agent=LinearAgent(environment=env,out_reward_window_td=out_reward_window,
                         reward_function="cum_return",sample_observations=32)

linear_agent.REINFORCE_fit(add_baseline=True)