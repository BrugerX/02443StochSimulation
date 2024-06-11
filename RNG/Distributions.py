from abc import ABC,abstractmethod
import RNG.RNG
import numpy as np
import RNG.DiscreteRV as drv
import RNG.ContinuousRV as crv

def expInverse(U,_lambda):
    return -np.log(U)/_lambda


class Distribution(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def getSample(self,n):
        return [0]

class Exponential(Distribution):

    def __init__(self,_lambda):
        self._lambda = _lambda

    def getSample(self, n = 1):
        return expInverse(np.random.uniform(0,1,n),self._lambda)

    def getSampleU(self, U_samples):
        return expInverse(U_samples, self._lambda)



class Gamma(Distribution):

    def __init__(self, _lambda,k):
        self._lambda = _lambda
        self.k = k

    def getSample(self, n=1):
        return [np.random.gamma(self.k,1/self._lambda,n)]

class ConstantDistribution(Distribution):

    def __init__(self,k):
        self.k = k

    def getSample(self,n = 1):
        return [self.k for _ in range(n)]

class Pareto(Distribution):

    def __init__(self,k,beta,null_lower = False):
        self.k = k
        self.beta = beta
        self.null_lower = null_lower

    def getSample(self,n = 1):
        U_samples = np.random.uniform(0,1,n)
        return crv.ParetoInverse(U_samples,self.k,self.beta,self.null_lower)



class MixtureModel(Distribution):

    """

    Distribution dict: probability x Distribution

    """
    def __init__(self,distributionTuple):
        self.X = np.array([p for (p,_) in distributionTuple])
        self.D = np.array([d for (_,d) in distributionTuple])

    def getSample(self,n = 1):
        U_samples = np.random.uniform(0,1,n)
        F = drv.crude_approx(self.X,U_samples)
        F_samples = []

        for fi in F:
            sub_sample = self.D[fi].getSample(1)
            F_samples.append(sub_sample)

        return np.array(F_samples).flatten()


    def getSampleU(self,U_samples):
        F = drv.crude_approx(self.X,U_samples)
        F_samples = []

        for fi in F:
            sub_sample = self.D[fi].getSample(1)
            F_samples.append(sub_sample)

        return np.array(F_samples).flatten()


class PredefinedModel(Distribution):

    def __init__(self,predefined_samples):
        self.predefined_samples = predefined_samples
        self.i = 0

    def getSample(self,n=1):
        sample = self.predefined_samples[self.i:self.i + n]
        self.i += n
        return sample