
from environments.e_greedy import DeepTradingEnvironment, LinearAgent,DeepAgentPytorch
import datetime
import numpy as np


out_reward_window=datetime.timedelta(days=1)
# parameters related to the transformation of data, this parameters govern an step before the algorithm
meta_parameters = {"in_bars_count": 32,
                   "out_reward_window":out_reward_window ,
                   "state_type":"in_window_out_window",
                   "asset_names":["asset_1","asset_2"],
                   "risk_aversion":1e4,
                   "include_previous_weights":False}

# parameters that are related to the objective/reward function construction
objective_parameters = {"percent_commission": .001,
                        }
print("===Meta Parameters===")
print(meta_parameters)
print("===Objective Parameters===")
print(objective_parameters)

assets_simulation_details={"asset_1":{"method":"GBM","sigma":.01,"mean":.02},
                    "asset_2":{"method":"GBM","sigma":.03,"mean":.18}}

env_min_vol=DeepTradingEnvironment.build_environment_from_simulated_assets(assets_simulation_details=assets_simulation_details,
                                                                     data_hash="simulation_gbm",
                                                                     meta_parameters=meta_parameters,
                                                                     objective_parameters=objective_parameters)
def create_environment():
    env=DeepTradingEnvironment.build_environment_from_simulated_assets(assets_simulation_details=assets_simulation_details,
                                                                     data_hash="simulation_gbm",
                                                                     meta_parameters=meta_parameters,
                                                                     objective_parameters=objective_parameters)
    return env


env=create_environment()
# env.state.reward_factory.ext_covariance=cov
linear_agent_min_vol=LinearAgent(environment=env,out_reward_window_td=out_reward_window,
                         reward_function="return_with_variance_risk",sample_observations=32)
linear_agent_min_vol.REINFORCE_fit(add_baseline=False,max_iterations=4000,plot_every=500)