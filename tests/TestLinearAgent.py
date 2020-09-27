
from lib.Environment import DeepTradingEnvironment, LinearAgent



import datetime

out_reward_window=datetime.timedelta(days=7)
# parameters related to the transformation of data, this parameters govern an step before the algorithm
meta_parameters = {"in_bars_count": 30,
                   "out_reward_window":out_reward_window ,
                   "state_type":"in_window_out_window"}

# parameters that are related to the objective/reward function construction
objective_parameters = {"percent_commission": .001}
print("===Meta Parameters===")
print(meta_parameters)
print("===Objective Parameters===")
print(objective_parameters)


assets_simulation_details={"asset_1":{"method":"GBM","sigma":.1,"mean":.1},
                    "asset_2":{"method":"GBM","sigma":.2,"mean":.2}}

# env=DeepTradingEnvironment.build_environment_from_simulated_assets(assets_simulation_details=assets_simulation_details,
#                                                                      data_hash="simulation_gbm",
#                                                                      meta_parameters=meta_parameters,
#                                                                      objective_parameters=objective_parameters)

env=DeepTradingEnvironment.build_environment_from_dirs_and_transform(
                                                                     data_hash="test_dirs",
                                                                     meta_parameters=meta_parameters,
                                                                     objective_parameters=objective_parameters)


linear_agent=LinearAgent(environment=env,out_reward_window_td=out_reward_window)
linear_agent.sample_env(observations=32)