import numpy as np

def controlVariable(X,Y,mu):
    c = -1 * np.cov(X, Y)[1, 0] / np.var(Y)
    Z = X + c * (Y - mu)
    return Z