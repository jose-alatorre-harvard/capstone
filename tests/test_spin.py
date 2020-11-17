
from lib.Environment2 import DeepTradingEnvironment

import numpy as np
import pandas as pd
import datetime
from spinup import vpg_pytorch
from lib.vpg_capstone import vpg as vpg_capstone
from lib.sac.sac import sac as sac_capstone
from lib.sac.core import MLPActorCritic as MLPActorCriticCapstone
from spinup import sac_pytorch

out_reward_window=datetime.timedelta(days=7)
meta_parameters = {"in_bars_count": 30,
                   "out_reward_window":out_reward_window ,
                   "state_type":"in_window_out_window",
                   "asset_names":["asset_1","asset_2"],
                   "include_previous_weights":False}

objective_parameters = {"percent_commission": .001,
                        "reward_function":"min_realized_variance"
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


#uses standard version of spinning-up
# vpg_pytorch(env_fn=env_fun,ac_kwargs={"hidden_sizes":(2,)},steps_per_epoch=32,epochs=2000)
#uses modified version of spinning-up
# vpg_capstone(env_fn=env_fun,ac_kwargs={"hidden_sizes":(2,)},steps_per_epoch=32,epochs=8000)

#SAV standard version of spinning-up

# sac_pytorch(env_fn=env_fun,ac_kwargs={"hidden_sizes":(2,)})

sac_capstone(env_fn=env_fun,actor_critic=MLPActorCriticCapstone,ac_kwargs={"hidden_sizes":(1,)},update_every=32,steps_per_epoch=32,epochs=2000,
             start_steps=100,update_after=32*5,alpha=.2, lr=1e-2)
