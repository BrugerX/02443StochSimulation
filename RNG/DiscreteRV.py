import numpy as np
import bisect

#k = number of options to choose from
def rejectionMethod(samples,X,k):
    c = np.max(X) + 0.0000001

    X_reject = []
    for idx,u1 in enumerate(samples):
        I = int(min(np.floor((k*u1)) + 1,k))
        pi = X[I]
        if(samples[idx - 1]<=pi/c):
            X_reject += [I]

    return X_reject

def crude_approx(X,samples):
    CDF_X = X.cumsum()

    X_approx = []
    for s in samples:
        X_approx += [bisect.bisect_right(CDF_X,s)]

    return X_approx