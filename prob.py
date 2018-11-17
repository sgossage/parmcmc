import numpy as np
import scipy
import time

from match.scripts import cmd
from match.MISTscripts.plotruns import *
from gausweight import *
import fileio as fio

def lnlike(obs, model):
    '''
    FROM MATCH README
    The quality of the fit is calculated using a Poisson maximum likelihood
    statistic, based on the Poisson equivalent of chi^2.
      2 m                                if (n=0)
      2 [ 0.001 + n * ln(n/0.001) - n ]  if (m<0.001)
      2 [ m + n * ln(n/m) - n ]          otherwise
    m=number of model points; n=number of observed points

    This statistic is based on the Poisson probability function:
       P =  (e ** -m) (m ** n) / (n!),
    Recalling that chi^2 is defined as -2lnP for a Gaussian distribution and
    equals zero where m=n, we treat the Poisson probability in the same
    manner to get the above formula.

    Returns
    -2lnP, P, sig i.e., (sqrt(-2lnP) * sign(n-m))
    '''
    
    # This is M_i:
    m = model

    # The data:
    n = obs

    # global -2lnP:
    m2lnP = 2. * (m + n * np.log(n / m) - n)

    
    smalln = np.abs(n) < 1e-10
    m2lnP[smalln] = 2. * m[smalln]

    # limiting m < 0.001 bin values to 0.001:
    smallm = (m < 0.001) & (n != 0)
    m2lnP[smallm] = 2. * (0.001 + n[smallm] * np.log(n[smallm] / 0.001) - n[smallm])
    
    # bring -2lnP back to ln P
    lnprob = -1 * np.sum(m2lnP) / 2

    return lnprob

# function that checks if x [float or int] is w/i range specified.
def inrange(x, lims):

    """
        Checks if x is in between lims (a 2-element array containing upper and lower 
    bounds of some range of numbers), this is noninclusive.
    """

    return max(lims) > x > min(lims)

def lnprior(theta, obsmass, lims):

    """
        Represents the prior distribution for parameters in the MCMC fit. This uses a 
    flat prior for a0 through a6 (the rotation disribution weights), and for the std. 
    deviation (sigma) and mean (mu) of the age distribution gaussian. The limits are 
    set to keep numbers sensible for the rotatation weights (positive, with some 
    upper bound). The sigma and mu are bounded according to the limits of our model 
    age grid structure (i.e., the finite age spacing means no sigma < 0.02, and model 
    space goes only from e.g., log Age = 8.5 to 9.5).

    """

    a0,a1,a2,a3,a4,a5,a6 = theta[:7]
    mu = theta[7]
    if len(theta) == 9:
        sigma = theta[8]
        if sigma < 0.02 or sigma > 1.0:
            return -np.inf
    else:
        sigma = 0.0
    
    #if (np.sum(theta[:7]) <= obsmass):
    #    valid = True
    #else:
    #    valid = False

    # flat prior, 0 if outside valid parameter range, or 1 if w/i valid range. 
    # (Recalculated for log-prob accordingly.)
    if inrange(a0, lims[0]) and inrange(a1, lims[1]) and inrange(a2, lims[2]) and inrange(a3, lims[3]) \
       and inrange(a4, lims[4]) and inrange(a5, lims[5]) and inrange(a6, lims[6]) and mu < 9.5 and \
       mu > 8.5:

        return 0.0

    return -np.inf

# Full Probability
def lnprob(theta, obs, model, lims):
    
    """
        Represents the full log-Probability (prior*likelihood). The "obs" is a 
    Hess diagram of the observed data under consideration. The "model" is an 
    array of model Hess diagrams, organized as an i x j matrix with i indexing 
    the rotation v/vc values, j indexing the grid points in age. The weights 
    are applied column-wise and row-wise appropriately, and all weighted model 
    Hess diagrams are then summed to form a composite model population Hess 
    diagram. The composite model is then compared to the data via our 
    likelihood calculation and evaluated.
    """
    #start = time.time()

    # current weights:
    #a0,a1,a2,a3,a4,a5,a6, mu, sigma = theta[]
    weights = theta[:7]#np.array([a0,a1,a2,a3,a4,a5,a6])
    mu = theta[7]
    if len(theta) == 9:
        sigma = theta[8]
    else:
        sigma = 0.0

    # models are formed here w/ weights:
    M = np.zeros(len(obs))

    # (we're fitting for the rotation weights of ea. model,
    # and the mean/std. dev. of a gaussian age distribution.
    age_range = np.arange(8.5,9.5,0.02)
    ageweights = gen_gaussweights(age_range, mu, sigma)
    # model[i] is the vector of all j Hess diagrams varying in age, and 
    # constant in the ith v/vc. I.e., model = model[i][j]...
    M += np.sum((10**weights[:, np.newaxis])*np.sum(ageweights[:,np.newaxis]*model, axis=1), axis=0)

    #print(max(M), max(obs))

    # rotation weight limes (these effectively scale population mass)
    #lims = [-1, np.log10(np.sum(obs))+3.5]

    obsmass = np.log10(np.sum(obs))
    # prior probability:
    lp = lnprior(theta, obsmass, lims)

    if not np.isfinite(lp):
        return -np.inf

    #end = time.time()
    #print("t = {:f} s".format(end - start))
    # full posterior probabliity, calls likelihood:
    return lp + lnlike(obs, M)

def linear_uncof(t, mode):

    """
        This converts log uncertainties (reported by default as things are written) 
    into linear uncertainties. Returns the + and - linear uncertainties. Expects an 
    array (3 elements) of 16th, 50th, and 84th percentiles from marginalized parameter 
    PDF.
    """

    #upuncs, louncs = [], [] 

    #for quants in t:
    #    upuncs.append(10**(t[1]+t[0]) - 10**t[0])
    #    louncs.append(10**t[0] - 10**(t[2]+t[0]))
    if mode == "+":
        return 10**(t[1]+t[0]) - 10**t[0]
    elif mode == "-":
        # 50th - 
        return 10**t[0] - 10**(t[0]-t[2])

    # return np.array(upuncs), np.array(louncs)

def get_results(samples):

    """
        Retrieves the 16th, 50th, and 84th percentile results from our final marginalized 
    distributions.
    """

    # Calculates the 16th, 50th, 84th percentiles for each parameter and then returns the 
    # 50th, 84th-50th, and 50th-16th:
    # tX = various v/vc weights
    # mu = mean of age spread gaussian
    # s = std. deviation of age spread gaussian
    try:
        t0, t1, t2, t3, t4, t5, t6, mu, s = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), 
                                               zip(*scipy.stats.scoreatpercentile(samples, [16, 50, 84], 
                                               axis=0)))
        # percentiles gathered into an array:
        percentiles = np.array([t0, t1, t2, t3, t4, t5, t6, mu])
    except ValueError:
        t0, t1, t2, t3, t4, t5, t6, mu = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), 
                                               zip(*scipy.stats.scoreatpercentile(samples, [16, 50, 84], 
                                               axis=0)))


    # percentiles gathered into an array:
        percentiles = np.array([t0, t1, t2, t3, t4, t5, t6, mu])

    # weights found at 50th percentile:
    log_weights = np.array([x[0] for x in percentiles])
    lin_weights = 10**log_weights

    # + uncertainties (from 84th percentile)
    upunc = np.array([x[1] for x in percentiles])
    lin_upunc = np.array([linear_uncof(x, "+") for x in percentiles])

    # - uncertainties (from 16th percentile)
    lounc = np.array([x[2] for x in percentiles])
    lin_lounc = np.array([linear_uncof(x, "-") for x in percentiles])

    # dictionaries of the uncertainties:
    log_err = {'higher': upunc, 'lower': lounc}
    lin_err = {'higher': lin_upunc, 'lower': lin_lounc}

    return log_weights, lin_weights, percentiles, log_err, lin_err

def genmod_agespread(cmddir, mass, agemu, agesig):

    ages = np.arange(8.50, 9.50, 0.02)
    ageweights = gen_gaussweights(ages, agemu, agesig)

    photbase = cmddir.split('/')[1]
    bf, av, agebin, logz, dmod = fio.parse_fname(photbase, mode="str")

    hess_arr = []
    for age in ages:
 
        a_cmd = cmd.CMD(fio.get_cmdf(cmddir, bf, age, logz, 0.0, av, dmod))        
        mock_hess = a_cmd.cmd['Nsim']
        mock_hess /= np.sum(mock_hess)
        #if age == ages[0]:
        #    composite_hess = mock_hess
        #else:
        #    composite_hess += mock_hess
        hess_arr.append(mock_hess)

    hess_arr = np.array(hess_arr)

    #print(len(hess_arr))
    #print(len(ageweights))
    composite_hess = np.sum(ageweights*hess_arr.T, axis=1)
    #print(len(composite_hess))    

    truth = np.sum(composite_hess)

    return mass*composite_hess

def genmod_vvcspread(cmddir, age, mass, vvcmu, vvcsig):

    vvcs = np.arange(0.0, 0.7, 0.1)
    mu, sig = 0.3, 0.05
    vvcweights = gen_gaussweights(vvcs, vvcmu, vvcsig)

    photbase = cmddir.split('/')[1]
    bf, av, agebin, logz, dmod = fio.parse_fname(photbase, mode="str")

    hess_arr = []
    for vvc in vvcs:
 
        a_cmd = cmd.CMD(fio.get_cmdf(cmddir, bf, age, logz, vvc, av, dmod))
        mock_hess = a_cmd.cmd['Nsim']
        mock_hess /= np.sum(mock_hess)
        #if age == ages[0]:
        #    composite_hess = mock_hess
        #else:
    #    composite_hess += mock_hess
        hess_arr.append(mock_hess)

    hess_arr = np.array(hess_arr)

    #print(len(hess_arr))
    #print(len(vvcweights))
    composite_hess = np.sum(vvcweights*hess_arr.T, axis=1)
    #print(len(composite_hess))
    truths = {}
    for i, vvc in enumerate(vvcs):
        truths[vvc] = mass*vvcweights[i]

    return mass*composite_hess, truths

def genmod_agevvcspread(cmddir, mass, agemu, agesig, vvcmu, vvcsig):

    ages = np.arange(8.50, 9.50, 0.02)
    #mu, sig = 9.00, 0.3
    ageweights = gen_gaussweights(ages, agemu, agesig)

    vvcs = np.arange(0.0, 0.7, 0.1)
    #mu, sig = 0.3, 0.2
    vvcweights = gen_gaussweights(vvcs, vvcmu, vvcsig)

    photbase = cmddir.split('/')[1]
    bf, av, agebin, logz, dmod = fio.parse_fname(photbase, mode="str")

    composite_hess = cmd.CMD(fio.get_cmdf(cmddir, bf, 9.00, logz, 0.0, av, dmod))
    composite_hess = np.zeros(len(composite_hess.cmd['Nsim']))

    vvc_pts = []
    for i, a_vvc in enumerate(vvcs):
        # step through in age, adding an age vector at each point in v/vc space
        age_vector = [] 
        for j, an_age in enumerate(ages):

            a_cmd = cmd.CMD(fio.get_cmdf(cmddir, bf, an_age, logz, a_vvc, av, dmod))
            model_hess = a_cmd.cmd['Nsim']
            model_hess /= np.sum(model_hess)

            age_vector.append(model_hess)

        vvc_pts.append(np.array(age_vector))

    model = np.array(vvc_pts)

    composite_hess += np.sum((vvcweights[:, np.newaxis])*np.sum(ageweights[:,np.newaxis]*model, axis=1), axis=0)

    #print(len(hess_arr))
    #print(len(vvcweights))
    #composite_hess = np.sum(vvcweights*hess_arr.T, axis=1)
    #print(len(composite_hess))
    truths = {}
    for i, vvc in enumerate(vvcs):
        truths[vvc] = mass*vvcweights[i]

    return mass*composite_hess, truths
