import numpy as np
"""
Server class without buffer or priority

"""
class BlockingServers:

    def __init__(self,service_dist,m):
        self.m = m
        self.S_t = service_dist
        self.serverSchedule = np.zeros(self.m)
        self.serviceTimes = []

    def getAvailableServerIdxs(self,t):
        return np.where(self.serverSchedule < t)[0]

    """
    @return 1 if unable to handle request else 0
    """
    def scheduleService(self,t_arrival):
        availableServers = self.getAvailableServerIdxs(t_arrival)

        if(len(availableServers) == 0):
            return 1 #Server unable to handle request
        else:
            serviceTime = self.S_t.getSample()[0]
            self.serviceTimes += [serviceTime] #Keep track of when it has servied for efficiency checking perhaps?
            self.serverSchedule[availableServers[0]] = t_arrival + serviceTime
            return 0

def getConfidenceInterval(subsamples,alpha = 0.05):
    n = len(subsamples)
    mean_hat = np.mean(subsamples)
    S_phi = np.std(subsamples)

    #We assume normal distribution
    if(alpha == 0.05):
        k = 1.96

    ci_constant = k*(S_phi/np.sqrt(n))
    return [mean_hat - ci_constant, mean_hat + ci_constant]


def getNrBlocked(sample):
    second_elements = [blocked for (t, blocked) in sample]
    return second_elements

def getSubProportions(subsamples):
    proportions_per_simulation = []

    for sample in subsamples:
        second_elements = getNrBlocked(sample)

        count_ones = second_elements.count(1)

        # Calculate the proportion of 1's
        total_elements = len(second_elements)
        proportions_per_simulation += [count_ones / total_elements]

    return proportions_per_simulation


def getSubMeans(subsamples):
    means = []

    for sample in subsamples:
        means += np.mean(sample)

    return means
