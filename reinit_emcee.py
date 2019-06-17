import numpy as np
from numpy.random import multivariate_normal
import matplotlib.pyplot as plt

def reinit_burn(pos, prob, threshold=50, pad=0.0):

    #print("prob ", prob)
    #print("percentile ", np.percentile(prob, threshold))
    # good walkers:
    good = prob > np.percentile(prob, threshold)

    #print("good i ", good)

    center = pos[good, :].mean(axis=0)

    # debugging
    #print("good pos ", pos[good, :])
    #print("good prob ", prob[good])

    Sigma = np.cov(pos[good, :].T)
    Sigma[np.diag_indices_from(Sigma)] += pad**2

    #print("Sigma ", Sigma)
    
    nwalkers = prob.shape[0]

    # draw new positions from Gaussian formed from 'good' walkers:
    new_pos = multivariate_normal(center, Sigma, size=nwalkers)

    return new_pos

def burn_emcee(sampler, new_pos, nburn=[16,32,64]):
    
    print("Culling walkers...")
    
    for k, burn in enumerate(nburn):
        print("Burn", burn)
        pos, prob, state = sampler.run_mcmc(new_pos, burn)
        new_pos = reinit_burn(pos, prob)
        sampler.reset()

    return new_pos, sampler
