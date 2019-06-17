import numpy as np
import sys
from scipy.integrate import quad

def gauss(x, m, s, a=1.0):

    return a*np.exp(- (x-m)**2 / (2*s**2))

def bigauss(x, m1, s1, m2, s2, a1=1.0, a2=1.0):

    return a1*np.exp(- (x-m1)**2 / (2*s1**2)) + a2*np.exp(- (x-m2)**2 / (2*s2**2))


def gen_gaussweights(v, m, s):

    N = len(v)

    # in the range of e.g. -10, 10...5 intervals = 5 weights
    # lims should be in order [lower, higher]
    #x = np.linspace(*lims, N+1)
    weights = np.zeros(N)

    if s > 0:
        for i in range(N):
            weights[i] = quad(gauss, v[i]-0.01, v[i]+0.01, args=(m,s))[0]
    else:
        # if std. dev is 0, give weight of 1 to closest Hess. If close to 
        # more than one, give equal weights to both (0.5 to both after 
        # normalization below).
        diff_arr = abs(v - m)
        idx = np.where(diff_arr == min(diff_arr))[0]
        weights[idx] += 1.0

    # normalize
    weights /= np.sum(weights)

    return weights

def gen_bigaussweights(v, m1, s1, m2, s2):

    N = len(v)                        

    # in the range of e.g. -10, 10...5 intervals = 5 weights
    # lims should be in order [lower, higher]
    #x = np.linspace(*lims, N+1)
    weights = np.zeros(N)

    if (s1 > 0) | (s2 > 0):
        for i in range(N):
            weights[i] = quad(bigauss, v[i]-0.01, v[i]+0.01, args=(m1, s1, m2, s2))[0]
    else:
        # if std. dev is 0, give weight of 1 to closest Hess. If close to 
        # more than one, give equal weights to both (0.5 to both after 
        # normalization below). Kinda clunky for bimodal gauss, but in 
        # practice won't be called (for now...).
        diff_arr = abs(v - m1)
        idx = np.where(diff_arr == min(diff_arr))[0]
        weights[idx] += 1.0

    # normalize
    weights /= np.sum(weights)

    return weights


if __name__ == '__main__':

    # deprecated calls
    m, s, lims, stepsize = sys.argv[1::]

    weights = gen_gaussweights(m, s, lims, stepsize)

    #print(weights)
