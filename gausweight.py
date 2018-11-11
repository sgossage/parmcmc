import numpy as np
import sys
from scipy.integrate import quad

def gauss(x, m, s, a=1.0):

    return a*np.exp(- (x-m)**2 / (2*s**2))

def gen_gaussweights(v, m, s):

    N = len(v) #int( round((max(lims) - min(lims)) / stepsize) )

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

    #for i in range(N):

        # generate 1000 samples in between x[i] and x[i+1]:
    #    samples = np.linspace(x[i], x[i+1], 1000)
        # ith weight is the integral of the gaussian function
        # in this range (like area under this section of 
        # gauss curve; approx. as summation):
    #    weights[i] = np.sum( gauss(samples, m, s) )

    # normalize
    weights /= np.sum(weights)
    #print(weights)
    #print(np.sum(weights))

    #assert np.sum(weights) == 1.0, "Gaussian weights do not sum to one."

    return weights

if __name__ == '__main__':

    m, s, lims, stepsize = sys.argv[1::]

    weights = gen_gaussweights(m, s, lims, stepsize)

    #print(weights)
