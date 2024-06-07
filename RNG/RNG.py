from abc import ABC,abstractmethod
from primefac import primefac

def find_a(M):
    a = 1
    x = 2
    for p in primefac(M):
        a *= p

    a *= x
    a += 1

    return a

class RNGToDist(ABC):

    @abstractmethod
    def transformNumber(self,n):
        pass

class DummyDistribution(RNGToDist):

    def __init__(self):
        pass

    def transformNumber(self,n):
        return n

class UniformDistribution(RNGToDist):

    def __init__(self,M):
        self.M = M

    def transformNumber(self,n):
        return n/self.M



class LCG:
    def __init__(self,a,M,c,seed,nrToGenerate = None,distribution = DummyDistribution()):
        self.a = a
        self.M = M
        self.c = c
        self.seed = seed

        self.xi_dict = {0:seed}
        self.i = 0

        if(nrToGenerate is None):
            self.nrToGenerate = self.M - 1
        else:
            self.nrToGenerate = nrToGenerate

        #If we don't have a distribution this will just be the dummy distribution
        self.distribution = distribution

    #Generates a U(0,1) variable
    def generateRN(self):
        xi = self.xi_dict[self.i]
        xii = (self.a*xi + self.c)%self.M
        self.i += 1
        self.xi_dict[self.i] = xii
        return self.distribution.transformNumber(xii)

    def __iter__(self):
        self.i = 0
        return self

    def __next__(self):
        if(self.i >= self.nrToGenerate):
            raise StopIteration

        return self.generateRN()

    def __len__(self):
        return self.nrToGenerate
