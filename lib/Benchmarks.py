
import numpy as np
import numpy.random as npr
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