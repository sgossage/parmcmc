import numpy as np
from numpy.random import multivariate_normal

def reinit_burn(pos, prob, threshold=50, pad=0.0):

    # good walkers:
    good = prob > np.percentile(prob, threshold)

    center = pos[good, :].mean(axis=0)
    Sigma = np.cov(pos[good, :].T)
    Sigma[np.diag_indices_from(Sigma)] += pad**2
    
    nwalkers = prob.shape[0]

    # draw new positions from Gaussian formed from 'good' walkers:
    new_pos = multivariate_normal(center, Sigma, size=nwalkers)

    return new_pos

def burn_emcee(sampler, new_pos, nburn=[16,32,64]):
    
    for k, burn in enumerate(nburn):
        pos, prob, state = sampler.run_mcmc(new_pos, burn)
        new_pos = reinit_burn(pos, prob)
        sampler.reset()

    return new_pos, sampler
