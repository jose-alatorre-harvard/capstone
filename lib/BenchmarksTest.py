from lib.Benchmarks import PortfolioBacktest
import numpy as np
import numpy.random as npr
import pandas as pd


class PortfolioBacktestTest():
    
    def __init__(self):
        self.asset_prices = self._generate_assets(
            start_date='2019-01-01', 
            end_date='2020-01-01', 
            assets=["EEMV", "LQD", "USMV", "EFAV", "MTUM", "UUP"]) 
        # equal weights
        self.weights = np.array([1/(len(self.asset_prices.columns)) for _ in range(len(self.asset_prices.columns))])
        self.commissions = None

    def _generate_assets(self, start_date, end_date, assets):
        """
        Generate Assets For Test Purposes
        :param start_date: start date of asset prices
        :param end_date: end date of asset prices
        :return: daily adj close of the assets in a dataframe
        """
        from pandas_datareader import data as web
        # df to store adj close prices of assets
        df = pd.DataFrame()
        for stock in assets:
            df[stock] = web.DataReader(stock,data_source='yahoo',start=start_date, end=end_date)['Adj Close']
        return df

    def test_portfolio_weights_backtest_statistics(self, verbose=True):
        pb = PortfolioBacktest(self.asset_prices, self.commissions)
        pb.portfolio_weights_backtest_statistics(self.weights, verbose)

    def test_rolling_sharpe_ratio(self, verbose=True):
        pb = PortfolioBacktest(self.asset_prices, self.commissions)
        pb.weights = self.weights
        rolling_sharpe_ratio = pb.build_rolling_sharpe_ratio(rolling_window=30)
        if verbose:
            print(rolling_sharpe_ratio.tail(10))

    def test_rolling_returns(self, verbose=True):
        pb = PortfolioBacktest(self.asset_prices, self.commissions)
        pb.weights = self.weights
        max_output = pb.build_rolling_max_return(rolling_window=30)
        min_output = pb.build_rolling_min_return(rolling_window=30)
        mean_output = pb.build_rolling_mean_return(rolling_window=30)

        if verbose:
            print("######")
            print("Rolling Max Returns Test")
            print(max_output.tail(10))
            print("######")
            print("Rolling Min Returns Test")
            print(min_output.tail(10))
            print("######")
            print("Rolling Mean Returns Test")
            print(mean_output.tail(10))

    def test_rolling_volatility(self, verbose=True):
        pb = PortfolioBacktest(self.asset_prices, self.commissions)
        pb.weights = self.weights
        rolling_volatility = pb.build_rolling_volatility(rolling_window=30)
        if verbose:
            print(rolling_volatility.tail(10))

    def test_all(self, verbose=True):
        print("Testing PortfolioBacktest Class ...")
        print("Testing ... portfolio_weights_backtest_statistics")
        self.test_portfolio_weights_backtest_statistics(verbose)
        print("Testing ... rolling_sharpe_ratio")
        self.test_rolling_sharpe_ratio(verbose)
        print("Testing ... rolling_returns")
        self.test_rolling_returns(verbose)
        print("Testing ... rolling_volatility")
        self.test_rolling_volatility(verbose)
        print("Testing Complete")
