
from lib.Environment2 import DeepTradingEnvironment
from lib.Environment2 import Actor
import numpy as np
import pandas as pd
import datetime
from spinup import vpg_pytorch

out_reward_window=datetime.timedelta(days=7)
meta_parameters = {"in_bars_count": 30,
                   "out_reward_window":out_reward_window ,
                   "state_type":"in_window_out_window",
                   "asset_names":["asset_1","asset_2"],
                   "include_previous_weights":False}

objective_parameters = {"percent_commission": .001,
                        "reward_function":"cum_return"
                        }
features=pd.read_parquet("/home/jose/code/capstone/temp_persisted_data/only_features_simulation_gbm")
forward_returns_dates=pd.read_parquet("/home/jose/code/capstone/temp_persisted_data/forward_return_dates_simulation_gbm")
forward_returns= pd.read_parquet("/home/jose/code/capstone/temp_persisted_data/only_forward_returns_simulation_gbm")
new_environment= DeepTradingEnvironment(objective_parameters=objective_parameters,meta_parameters=meta_parameters,
                                        features=features,
                                        forward_returns=forward_returns,
                                        forward_returns_dates=forward_returns_dates)

obs, reward, done, info=new_environment.step(action=np.array([.5,.5]))

env_fun =lambda : DeepTradingEnvironment(objective_parameters=objective_parameters,meta_parameters=meta_parameters,
                                        features=features,
                                        forward_returns=forward_returns,
                                        forward_returns_dates=forward_returns_dates)

