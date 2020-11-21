
from environments.e_greedy import DeepTradingEnvironment, LinearAgent,DeepAgentPytorch
import datetime
import numpy as np


out_reward_window=datetime.timedelta(days=7)
# parameters related to the transformation of data, this parameters govern an step before the algorithm
meta_parameters = {"in_bars_count": 30,
                   "out_reward_window":out_reward_window ,
                   "state_type":"in_window_out_window",
                   "asset_names":["asset_1","asset_2"],
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

env=DeepTradingEnvironment.build_environment_from_simulated_assets(assets_simulation_details=assets_simulation_details,
                                                                     data_hash="simulation_gbm",
                                                                     meta_parameters=meta_parameters,
                                                                     objective_parameters=objective_parameters)

# env=DeepTradingEnvironment.build_environment_from_dirs_and_transform(
#                                                                      data_hash="test_dirs",
#                                                                      meta_parameters=meta_parameters,
#                                                                      objective_parameters=objective_parameters)

#reward_function= "cum_return" , "max_sharpe"

cov=np.array([[assets_simulation_details["asset_1"]["sigma"]**2,0],[0,assets_simulation_details["asset_2"]["sigma"]**2]])
mus=np.array([assets_simulation_details["asset_1"]["mean"],assets_simulation_details["asset_2"]["mean"]])

from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.plotting import plot_efficient_frontier
from pypfopt.cla import CLA
ef = EfficientFrontier(mus, cov)
# weights = ef.max_sharpe(risk_free_rate=0)
weights = ef.min_volatility()

x=np.array(list(weights.values())).reshape(-1,1)
p_vol=np.sqrt(np.matmul(np.matmul(x.T,cov),x))
p_sharpe=np.matmul(x.T,mus)/p_vol

deep_agent=DeepAgentPytorch(environment=env,out_reward_window_td=out_reward_window,pre_sample=True,
                         reward_function="min_vol",sample_observations=64)


deep_agent.set_plot_weights(weights=np.array([0,1]), benchmark_G=assets_simulation_details["asset_2"]["mean"])
# deep_agent.set_plot_weights(weights=np.array(list(weights.values())),
#                               benchmark_G=-p_vol.ravel()[0])

deep_agent.ACTOR_CRITIC_fit()
deep_agent.REINFORCE_fit()


linear_agent=LinearAgent(environment=env,out_reward_window_td=out_reward_window,
                         reward_function="cum_return",sample_observations=32)

# cla=CLA(mus,cov)
# weights=cla.max_sharpe()
#max return all weights should go to asset with higher mean
# linear_agent.set_plot_weights(weights=np.array([0,1]), benchmark_G=assets_simulation_details["asset_2"]["mean"])

#min vol weithts



linear_agent.set_plot_weights(weights=np.array(list(weights.values())),
                              benchmark_G=-p_vol.ravel()[0])

# # linear_agent.set_plot_weights(weights=np.array(list(weights.values())),
# #                               benchmark_G=p_sharpe.ravel()[0])
#
# linear_agent.REINFORCE_fit(add_baseline=True,plot_gradients=True)
# # linear_agent.REINFORCE_refactor_fid()

linear_agent.ACTOR_CRITIC_FIT(use_traces=True)
