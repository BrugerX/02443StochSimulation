import numpy as np

#null_lower = false => x>=beta else x>=0
def ParetoInverse(samples,k,beta,null_lower = False):
    if(null_lower):
        return beta*(samples**(-1/k))

    return beta*(samples**(-1/k)-1)

def expInverse(U,_lambda):
    return -np.log(U)/_lambda