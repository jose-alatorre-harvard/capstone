
from lib.Environment import DeepTradingEnvironment, LinearAgent



import datetime
import numpy as np
from tqdm import tqdm

out_reward_window=datetime.timedelta(days=7)
# parameters related to the transformation of data, this parameters govern an step before the algorithm
meta_parameters = {"in_bars_count": 30,
                   "out_reward_window":out_reward_window ,
                   "state_type":"in_window_out_window"}

# parameters that are related to the objective/reward function construction
objective_parameters = {"percent_commission": .001,
                        }
print("===Meta Parameters===")
print(meta_parameters)
print("===Objective Parameters===")
print(objective_parameters)


assets_simulation_details={"asset_1":{"method":"GBM","sigma":.1,"mean":.1},
                    "asset_2":{"method":"GBM","sigma":.1,"mean":-.1}}

env=DeepTradingEnvironment.build_environment_from_simulated_assets(assets_simulation_details=assets_simulation_details,
                                                                     data_hash="simulation_gbm",
                                                                     meta_parameters=meta_parameters,
                                                                     objective_parameters=objective_parameters)

# env=DeepTradingEnvironment.build_environment_from_dirs_and_transform(
#                                                                      data_hash="test_dirs",
#                                                                      meta_parameters=meta_parameters,
#                                                                      objective_parameters=objective_parameters)

linear_agent=LinearAgent(environment=env,out_reward_window_td=out_reward_window,
                         reward_function="cum_return")

cov=np.array([[assets_simulation_details["asset_1"]["sigma"],0],[0,assets_simulation_details["asset_2"]["sigma"]]])
mus=np.array([assets_simulation_details["asset_1"]["mean"],assets_simulation_details["asset_2"]["mean"]])

#max return all weights should go to asset with higher mean
linear_agent.set_plot_weights(weights=np.array([0,1]))

linear_agent.REINFORCE_linear_fit()


