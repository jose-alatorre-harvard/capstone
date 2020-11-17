
import numpy as np
import numpy.random as npr


#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Asset Simulation >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


class GBMBasketSimulation():


    def __init__(self,n_assets,means,sigmas,correlation_matrix=None):
        """

        :param n_assets:
        :param correlation_matrix:
        """
        self.n_assets=n_assets
        self.means=means
        self.sigmas=sigmas

        if correlation_matrix is not None:
            self.correlation_matrix = correlation_matrix
        else:
            self.correlation_matrix=np.eye(self.n_assets)


    def simulate_returns(self,dt_in_years,n_returns):
        T = dt_in_years
        I = n_returns
        M=1
        cho_mat = np.linalg.cholesky(self.correlation_matrix)

        ran_num = npr.standard_normal((2, M + 1, I))

        returns = np.exp((self.means - 0.5 * self.sigmas ** 2) * T + self.sigmas * np.sqrt(T) * npr.standard_normal(I))



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

    def __init__(self, asset_prices, commission):
        pass

    def build_portfolio_backtest(self, weights):
        """
        builds backtest of the selected weights.  important to notice that weights correspond to end of period
        :return:
        """
        pass

    def create_rolling_high_sharpe(self, rolling_window):
        """

        :param rolling_window: datetime.timedelta
        :return: pd.DataFrame rolling_weights
        """
        pass

    def create_rolling_min_vol(self, rolling_window):
        """

               :param rolling_window:datetime.timedelta
               :return: pd.DataFrame rolling_weights
               """
        pass

    def create_rolling_max_return(self, rolling_window):
        """

           :param rolling_window:datetime.timedelta
           :return: pd.DataFrame rolling_weights
       """
        pass

    def plot_asset_turnover(self, historical_weights):
        """
        plots assets turnover. Idea: Box plot of each asset weights with time as hue.
        :param historical_weights:
        :return:
        """

    def plot_efficient_frontier(self, expected_returns, covariance, portfolios_weights):
        """
        plots and efficient frontier and the location of the portfolios.
        :param expected_returns:
        :param covariance:
        :param portfolios_weights: pd.DataFrame each row is a portfolio and columns are the asset weights
        :return:
        """