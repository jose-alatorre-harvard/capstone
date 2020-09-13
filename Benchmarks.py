
import numpy as np
import numpy.random as npr
class SimulatedAsset:


    def __init__ (self,sigma,mean):

        self.sigma=sigma
        self.mean=mean

    def simulate_returns(self,method,*args,**kwargs):
        """
        Factory for simulated returns
        :param method:
        :param args:
        :param kwargs:
        :return:
        """
        if method=="GMB":
            returns=self.simulate_returns(**kwargs)
        else:
            raise NotImplementedError

    def simulate_returns_GBM(self,time_in_years,n_returns):





        T = time_in_years
        I = n_returns
        returns= np.exp((self.mean - 0.5 * self.sigma ** 2) * T + self.sigma * np.sqrt(T) * npr.standard_normal(I))

        return returns



