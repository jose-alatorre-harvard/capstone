from lib.BenchmarkUtilities import *
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as npr
import pandas as pd
import seaborn as sns
sns.set_style("whitegrid")

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Asset Simulation >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

class SimulatedAsset:

    def simulate_returns(self,method,*args,**kwargs):
        """
        Factory for simulated returns
        :param method:
        :param args:
        :param kwargs:
        :return:
        """
        if method=="GBM":
            returns=self.simulate_returns_GBM(**kwargs)
        else:
            raise NotImplementedError

        return returns

    def simulate_returns_GBM(self,time_in_years,n_returns,sigma,mean):

        T = time_in_years
        I = n_returns
        returns= np.exp((mean - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * npr.standard_normal(I))
        return returns


    def simulate_returns_GARCH(self,time_in_years,n_returns,sigma,mean):

        T=time_in_years
        vol = sigma * np.sqrt(T)
        alpha = .06
        beta = .92
        w = vol * vol * (1 - alpha - beta)

        variances = []
        noises = []
        returns=[]
        for i in range(n_returns):

            if i > 0:
                noises.append(np.random.normal(loc=0, scale=np.sqrt(variances[i - 1])))
                v = w + alpha * (noises[i - 1] ** 2) + beta * variances[i - 1]
            else:
                v = w

            variances.append(v)
            r=np.exp((mean - 0.5 * variances[i] ** 2) * T +np.sqrt(variances[i])* npr.standard_normal(n_returns))

            returns.append(r)

        return returns


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Portfolio Construction >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

class PortfolioBacktest:

    def __init__(self, asset_prices, commissions):
        """
        :param: asset_prices: pandas.DataFrame
        :commissions
        """
        self.asset_prices = asset_prices
        self.commissions = commissions
        self.weights = None

    def build_portfolio_backtest(self, weights):
        """
        builds backtest of the selected weights.  
        important to notice that weights correspond to end of period
        :return:
        """
        self.weights = weights
        return portfolio_weights_backtest_statistics(weights)

    def portfolio_weights_backtest_statistics(self, weights, verbose=False):
        """
        compute statistics of the backtest portfolio allocation based on weights
        :param: weights: (list) list of weights 
        """
        returns = compute_daily_returns(self.asset_prices)
        cov_matrix = compute_covariance_matrix(returns)
        portfolio_variance = compute_expected_portfolio_variance(cov_matrix, weights)
        portfolio_volatility = compute_expected_portfolio_volatility(portfolio_variance)
        portfolio_annual_returns = compute_annual_return(returns, weights)
        if verbose:
            print("Portfolio Volatility: {:.2%}".format(portfolio_volatility))
            print("Portfolio Annual Returns: {:.2%}".format(portfolio_annual_returns))
        return portfolio_volatility, portfolio_annual_returns

    def build_rolling_sharpe_ratio(self, rolling_window):
        """
        Builds Rolling Sharpe Ratio Based on Asset Prices
        :param rolling_window: datetime.timedelta
        :return: pd.DataFrame rolling_weights
        """
        if (self.weights is None): raise Exception()

        # get normalized returns from adj returns 
        asset_norm_returns = get_normalized_returns(self.asset_prices)
        asset_allocation_df = build_asset_allocation_df(asset_norm_returns, self.weights)
        rolling_sharpe_ratio = compute_rolling_sharpe_ratio(asset_allocation_df, rolling_window)
        return rolling_sharpe_ratio

    def build_rolling_max_return(self, rolling_window):
        """
        Builds Rolling Max Returns Based on Asset Prices
        :param rolling_window: datetime.timedelta
        :return: pd.DataFrame rolling_weights
       """
        if (self.weights is None): raise Exception()

        # get normalized returns from adj returns 
        asset_norm_returns = get_normalized_returns(self.asset_prices)
        asset_allocation_df = build_asset_allocation_df(asset_norm_returns, self.weights)
        rolling_max_returns = compute_rolling_returns(asset_allocation_df, 'max', rolling_window)
        return rolling_max_returns

    def build_rolling_min_return(self, rolling_window):
        """
        Builds Rolling Min Returns Based on Asset Prices
        :param rolling_window: datetime.timedelta
        :return: pd.DataFrame rolling_weights
       """
        if (self.weights is None): raise Exception()

        # get normalized returns from adj returns 
        asset_norm_returns = get_normalized_returns(self.asset_prices)
        asset_allocation_df = build_asset_allocation_df(asset_norm_returns, self.weights)
        rolling_min_returns = compute_rolling_returns(asset_allocation_df, 'min', rolling_window)
        return rolling_min_returns

    def build_rolling_mean_return(self, rolling_window):
        """
        Builds Rolling Mean Returns Based on Asset Prices
        :param rolling_window: datetime.timedelta
        :return: pd.DataFrame rolling_weights
       """
        if (self.weights is None): raise Exception()

        # get normalized returns from adj returns 
        asset_norm_returns = get_normalized_returns(self.asset_prices)
        asset_allocation_df = build_asset_allocation_df(asset_norm_returns, self.weights)
        rolling_mean_returns = compute_rolling_returns(asset_allocation_df, 'mean', rolling_window)
        return rolling_mean_returns

    def build_rolling_volatility(self, rolling_window):
        """
        Builds Rolling Volatility Based on Asset Prices
        :param rolling_window: datetime.timedelta
        :return: pd.DataFrame rolling_weights
       """
        if (self.weights is None): raise Exception()

        # get normalized returns from adj returns 
        asset_norm_returns = get_normalized_returns(self.asset_prices)
        asset_allocation_df = build_asset_allocation_df(asset_norm_returns, self.weights)
        rolling_volatility = compute_rolling_volatility(asset_allocation_df, rolling_window)
        return rolling_volatility

    # TODO - works in notebook, implement saving image to path
    def plot_rolling_sharpe_ratio(self, rolling_sharpe_ratio):
        rolling_sharpe_ratio[rolling_sharpe_ratio['rolling_sharpe_ratio'] > 0].rolling_sharpe_ratio.plot(style='-', lw=3, color='orange', 
                                            label='Sharpe Ratio', figsize = (10,7)).axhline(y = 1.6, color = "blue", lw = 3,linestyle = '--')
        plt.ylabel('Sharpe Ratio')
        plt.legend(loc='best')
        plt.title('Rolling Sharpe ratio')
        plt.show()
