#!/usr/bin/env python
import numpy as np
import scipy
import emcee
import matplotlib as mpl
mpl.use('Agg')

import matplotlib.pyplot as plt
import glob
import sys
import os
import seaborn as sns

from match.scripts import cmd
from match.MISTscripts.plotruns import *
from reinit_emcee import *
from gausweight import *
from random import gammavariate

import graphics as gfx
import prob as pr
from prob import lnprob
import fileio as fio

#from functools import partial


#lnprob_part = partial(lnprob, kwargs = {"obs":obs, "model":model, "lims":lims})

#pool = emcee.utils.MPIPool(debug=False, loadbalance=True)
#if not pool.is_master():
    # Wait for instructions from the master process.
#    pool.wait()
#    sys.exit(0)

# right now 4 inputs available/required: cmddir (where .cmd files live), ncpu (num cpus for multiproc), 
# svdir (directory where plots are saved to), mode (either specifying mock or a run on observations).

# cmddir has format output/photbasename/mcmc/

if __name__ == "__main__":

#=================================================================================================================================
# Initialization
    print("Running with {:s} CPUs...".format(sys.argv[2]))
    print("Getting files...")

    # cuts te given directory containing the .cmd files to get the photometry file base (system dep, kinda bad rule for this).
    photbase = sys.argv[1].split('/')[1]

    # The binary fraction, etc. are taken from the photometry filename:
    bf, av, agebin, logz, dmod = fio.parse_fname(photbase, mode="str")
    av = format(float(av),'.2f')

    # move down a grid point if given age is off age grid (not even)
    agebinf = np.array(list(map(float, agebin)))
    if not (agebinf[0]*100) % 2 == 0:
        agebinf = agebinf - 0.01
        agebin = list(map(str, agebinf))

    print(bf, av, agebin, logz, dmod)
    print(type(bf), type(av), type(agebin[0]), type(agebin[1]), type(logz), type(dmod))
    
    # where the .cmd files are located:
    cmddir = sys.argv[1]

    # number of cpus, for multiprocessing:
    ncpu = int(sys.argv[2])

    # where plots will be saved:
    svdir = sys.argv[3]

    # if mode is "mock", a synthetic cluster will be created and used as data (for testing); uses the given vvcsig, etc. to generate.
    # So, "mock" will create perfect conditions for a test -- the models should recover the data more or less perfectly.
    # if mode is "mock-sigtau", a synthetic cluster with just an age distribution will be created, despite model parameter limits.
    # if mode is "mock-sigom", '                             ' a rotation distribution will be created, '                        '.
    # if mode is "mock-full", '                              ' everything will be created, despite model parameter limits.
    # if mode is "obs", a Hess diagram will be made from list of observed magnitudes (i.e., for a run using real data).
    mode = sys.argv[4]

    # MATCH style filter names:
    vfilter = sys.argv[5]
    ifilter = sys.argv[6]
    filters = [vfilter, ifilter]

    # Controls for age and v/vc distributions (agesig = 0.0 means no age spread/vvcsig = 0.0 means no v/vc spread) that the models 
    # are allowed to take on. I.e., specifies the scenario for the models to test.If mode is "mock", the mock data will also use 
    # these values and will match the scenario being considered by the models.
    agemu = round(float(sys.argv[7]), 2)
    agesig = round(float(sys.argv[8]), 2)
    vvcmu = round(float(sys.argv[9]), 1)
    vvcsig = round(float(sys.argv[10]), 1)

    # sets defaults for the mock data generation, given a specified mode of operation:
    # E.g., defaults used in previous tests were 9.00, 0.3, 0.3, 0.2.
    default_agemu = 9.00
    default_agesig = 0.05
    default_vvcmu = 0.4
    default_vvcsig = 0.2

    if mode == "mock-sigtau":
        # no rotation distribution:
        mockd_agemu, mockd_agesig, mockd_vvcmu1, mockd_vvcsig1 = [agemu, default_agesig, 0.0, 0.0]
    elif mode == "mock-sigom":
        # no age distribution
        mockd_agemu, mockd_agesig, mockd_vvcmu1, mockd_vvcsig1 = [agemu, 0.0, default_vvcmu, default_vvcsig]
    elif mode == "mock-bisigom":
        # no age distribution, bimodal distribution of rotation rates
        mockd_agemu, mockd_agesig, mockd_vvcmu1, mockd_vvcsig1, mockd_vvcmu2, mockd_vvcsig2 = [agemu, 0.0, 0.1, 0.1, 0.7, 0.2]
    elif mode == "mock-full":
        mockd_agemu, mockd_agesig, mockd_vvcmu1, mockd_vvcsig1 = [agemu, default_agesig, default_vvcmu, default_vvcsig]
    elif mode == "mock-fullbi":
        mockd_agemu, mockd_agesig, mockd_vvcmu1, mockd_vvcsig1, mockd_vvcmu2, mockd_vvcsig2 = [agemu, default_agesig, 0.1, 0.1, 0.7, 0.2]
    else:
        # here's the case of "mock", the mock data will perfectly match the scenario proposed for the models to explore:
        mockd_agemu, mockd_agesig, mockd_vvcmu1, mockd_vvcsig1 = [agemu, agesig, vvcmu, vvcsig]

#=================================================================================================================================
# Model generation for mock data, or else assigning "observations" to the given observed data.
# Truths are set here as well.

    # rotation distribution allowed in model search:
    if vvcsig > 0.0:
        Nrot = 10                                       # 10 is the current max, will create models @ 0.0, 0.1, ..., 0.9
    # rotation distribution disallowed in model search:
    elif (vvcsig == 0.0) & (agesig > 0.0):
        Nrot = 1                                        # Will only consider models @ om/omc = 0.0 when set to 1.

    print(Nrot)

    vvclim = round(Nrot / 10.0, 1)
    vvc_range = np.arange(0.0, vvclim, 0.1)
    age_range = np.arange(7.9, 9.5, 0.02) # lower lim was 8.5
    # default, dummy truths; will be reassigned:
    truths = {rot:1e-11 for rot in vvc_range}
    mass = 5E4                                          # mock cluster "mass" or total counts; 1e6 was default

    # Generates mock data from the model library to use instead of observed data if told to:
    if "mock" in mode:
        # case of no rotation distribution:
        if mockd_vvcsig1 == 0.0:
            # and no age distribution (would be an SSP):
            if mockd_agesig == 0.0:
                obscmd = cmd.CMD(fio.get_cmdf(cmddir, bf, mockd_agemu, logz, mockd_vvcmu1, av, dmod))
                obs = obscmd.cmd['Nobs']
            # age distribution, no rotation distribution, P(sigtau); in practice will trigger so long as mode is mock-sigtau and agesig !=0.0:
            else:
                print("GENERATING MOCK DATA WITH AGE SPREAD...")
                obs = pr.genmod_agespread(cmddir, mass=mass, agemu=mockd_agemu, agesig=mockd_agesig)

            obsweight = np.sum(obs)
            # using lower limit of weight search prior for truths of "zero" components.
            truths = {rot:max([0.0, np.log10(obsweight*1e-4)]) for rot in vvc_range}
            truths[round(float(mockd_vvcmu1), 1)] = obsweight
        
        # case allowing a rotation distribution:
        else:
            # and no age distribution P(sigom):
            if mockd_agesig == 0.0:
                print("GENERATING MOCK DATA WITH ROTATION SPREAD...")
       	       	#print(mockd_agemu)
       	       	#print(mockd_agesig)
       	       	#print(mockd_vvcmu)
                #print(mockd_vvcsig)
                if mode == "mock-bisigom":
                    obs, truths = pr.genmod_bivvcspread(cmddir, mockd_agemu, mass=mass, vvcmu1=mockd_vvcmu1, vvcsig1=mockd_vvcsig1, vvcmu2=mockd_vvcmu2, vvcsig2=mockd_vvcsig2, vvclim=1.0)                    
                else:
                    obs, truths = pr.genmod_vvcspread(cmddir, mockd_agemu, mass=mass, vvcmu=mockd_vvcmu1, vvcsig=mockd_vvcsig1, vvclim=1.0)
                print(truths)
                # Don't use full truth range of mock data's vvc distribution if fitting age spread model to pure rotation spread mock 
                # data b/c model only knows one rotation rate exists:
                if (vvcsig == 0.0) & (agesig > 0.0):
                    print("correcting truths...")
                    truths = {vvcmu:truths[vvcmu]}
                    print(truths)

            # allowing both an age and rotation rate distribution, P(sigom, sigtau); will trigger if vvcsig and agesig != 0 
            # and mode != "mock-sigom" or "mock-sigtau":        
            else:
                print("GENERATING MOCK DATA WITH AGE AND ROTATION SPREAD...")
                if mode == "mock-fullbi":
                    obs, truths = pr.genmod_agebivvcspread(cmddir, mass=mass, agemu=mockd_agemu, agesig=mockd_agesig, vvcmu1=mockd_vvcmu1, vvcsig1=mockd_vvcsig1, vvcmu2=mockd_vvcmu2, vvcsig2=mockd_vvcsig2, vvclim=1.0)
                else:
                    obs, truths = pr.genmod_agevvcspread(cmddir, mass=mass, agemu=mockd_agemu, agesig=mockd_agesig, vvcmu=mockd_vvcmu1, vvcsig=mockd_vvcsig1, vvclim=1.0)

        obsweight = mass

    # or else just use the observed Hess diagram values; don't create mock data from models.
    # truth is "unknown" here for the vvc weights; each is set to the total observed counts.
    elif mode == "obs":
        obscmd = cmd.CMD(fio.get_cmdf(cmddir, bf, agemu, logz, vvcmu, av, dmod))
        obs = obscmd.cmd['Nobs']
        obsweight = np.sum(obs)
        truths = {x: obsweight for x in truths}

    elif mode == "read-in":
        obscmd = np.genfromtxt(cmddir)
        obscmd = obscmd.T
        obs = obscmd[2]
        obsweight = np.sum(obs)
        truths = {x:obsweight for x in truths}
        cmddir = os.path.join(*cmddir.split('/')[:-2])

    # finalize format of truth values:
    lin_truths = np.array(list(truths.values()))
    truths = np.log10(lin_truths)
    if agesig == 0.0:
        truths = np.append(truths, agemu)
        lin_truths = np.append(lin_truths, 10**agemu)
        ndim = Nrot+1    
    else:
        truths = np.append(truths, [agemu, agesig])
        lin_truths = np.append(lin_truths, [10**agemu, 10**agesig])
        ndim = Nrot + 2

#=================================================================================================================================
# Build grid of models from which emcee will draw to do the data-model fit.

    model, dinds = fio.gather_models(cmddir, bf, age_range, logz, vvc_range, av, dmod)

#=================================================================================================================================
# Perform MCMC algorithm to attain best-fit parameters.

    # 9 dimensions (7 rotation rates, age, age std. deviation), 
    # however many walkers, number of iterations (1024, 512 seems to work well):
    minsteps = 32
    nsteps = 1024 #1024, 512
    # 32 steps minimum; the number of emcee culling steps is 16 minimum, so this ensures more 
    # actual steps than culling steps. Number of culling steps could be reduced below, etc. if desired.
    nsteps = max(minsteps, nsteps)
    nwalkers = nsteps*2
    # cut out first half of iterations as burn-in
    burn = -int(nsteps/2)

    # Setting walkers up on hyperplane of solutions for the rotation weights using Dirichlet disribution.
    print("Initializing walker positions...")
    rotw = np.array([])
    #sample = np.array([-1.]*Nrot) 
    #print("obsweight = ", obsweight)
    for i in range(nwalkers):
        # while not acceptably near the total observed weight...
        # (all(sample >= obsweight/20.))
        sample = np.array([-1.]*Nrot)
        while not (((all(sample >= obsweight/20.)) & (all(sample <= obsweight)))):
            # ensuring that all v/vc weights start off on plane wherein the sum of weights
            # is near the actual total 'observed' weight. 
            params = np.ones(Nrot)
            sample = np.array([gammavariate(a,1) for a in params])
            sample = np.array([v/sum(sample) for v in sample])
            sample = sample*obsweight
        if (i == 0):
            # add valid position to rotw array    
            rotw = np.log10(sample)
        else:
            #print(sample)
            # stack subsequent initial positions vertically
            rotw = np.vstack((rotw, np.log10(sample)))

    print("Initial positions for v/vc weights found...")
    print("Assigning remaining initial positions...")
    pos = []
    for i in range(nwalkers):
        posi = rotw[i]
        # initial age positions randomized according to uniform dist. about supplied mean age.
        posi = np.append(posi, agemu + np.random.uniform(-0.2, 0.2, 1))
        if ndim == Nrot+2:
            # initial age gaussian std. dev. randomized according to uniform dist. about supplied sigma.
            #if agesig > 0.0:
            posi = np.append(posi, agesig + np.random.uniform(-0.05, 0.05, 1))
            #else:
            #    posi = np.append(posi, 0.05 + np.random.uniform(-0.05, 0.05, 1))

        pos.append(posi)
    print(pos[0])
    print("Sample sum of v/vc weights (log10) = ", np.log10(np.sum(10**pos[0][:Nrot])))

    # prior distribution limits for v/vc weights (assuming flat priors).
    lims = np.array([[max([0.0, np.log10(obsweight*1e-4)]), np.log10(obsweight*1.1)] for truth in truths[:Nrot]]) # was log10(obs) +/- 2 dex

    # experimenting//checking if forcing rot dist on ngc 2249 reduces spread, it does, but not much at these levels
    #lims[0] = [2.18, np.log10(obsweight*1.1)]
    #lims[4] = [2.18, np.log10(obsweight*1.1)]
    #lims[8] = [2.18, np.log10(obsweight*1.1)]

    print(truths)
    print(lin_truths)
    print(lims)
    print(len(lims))
    print("NUMBER OF DIMENSIONS: ", ndim)
    #sys.exit()

    # the affine-invariant emcee sampler:
    print("Setting up MCMC sampler...")
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, kwargs = {"obs":obs, "model":model, "lims":lims, 
                                                                      "dinds":dinds, "vvc_range":vvc_range, "age_range":age_range}, threads=ncpu)
    #sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_part, pool=pool)

    print("Running MCMC...")
    # run mcmc:
    print("Thinning chains...")
    # creates an array of culling steps starting at minsteps/2, up to nsteps/2
    emcee_burn_steps = [int(minsteps*2**(x-1)) for x in range(int( (np.log(nsteps) - np.log(minsteps))/np.log(2) + 1))]
    new_pos, sampler = burn_emcee(sampler, pos, emcee_burn_steps)#16, 32, 64, 128, 256, 512, 1024])
    print("Doing full run...")
    pos, prob, state = sampler.run_mcmc(new_pos, nsteps)
#    pool.close()

#=================================================================================================================================
# Plot the results.

    print("Plotting results...")
    gfx.chain_plot(nwalkers, ndim, sampler, cmddir = cmddir, vvclim=vvclim, svdir=svdir, truths=truths, lintruths=lin_truths, burn=burn)

    print("Mean acceptance fraction: {0:.3f}"
                    .format(np.mean(sampler.acceptance_fraction)))

    # print(sampler.acor)
    try:
        print(sampler.get_autocorr_time())
    except Exception as e:
        pass

    # cut out burn-in phase; using last 1k steps as non-burn-in.
    samples = sampler.chain[:, burn:, :].reshape((-1, ndim))

    log_weights, lin_weights, percentiles, log_err, lin_err = pr.get_results(samples)

    # plot of best solutions (weights) vs. v/vc:
    f,ax=plt.subplots(1,1,figsize=(16,9))
    
    print(log_weights[:Nrot])
    print(log_err['lower'][:Nrot])
    print(log_err['higher'][:Nrot])

    ax.errorbar(vvc_range, log_weights[:Nrot], yerr=[log_err['lower'][:Nrot], log_err['higher'][:Nrot]], c='r', ls='--')
    ax.plot(vvc_range, truths[:Nrot], c='k')
    ax.set_xlabel(r'$\Omega/\Omega_c$', size=24)
    ax.set_ylabel('log Rel. Star Weight', size=24)
    f.savefig(os.path.join(cmddir, svdir, 'logrelweight_vs_vvc.png'))

    f,ax=plt.subplots(1,1,figsize=(16,9))
    ax.errorbar(vvc_range, lin_weights[:Nrot], yerr=[lin_err['lower'][:Nrot], lin_err['higher'][:Nrot]], c='r', ls='--')
    ax.plot(vvc_range, lin_truths[:Nrot], c='k')
    ax.set_xlabel(r'$\Omega/\Omega_c$', size=24)
    ax.set_ylabel('Rel. Star Weight', size=24)
    f.savefig(os.path.join(cmddir, svdir, 'relweight_vs_vvc.png'))

    _ = gfx.plot_random_weights(sampler, nsteps, ndim, log_weights, log_err, cmddir, vvclim, log=True, svdir=svdir, truths=truths, burn=burn)
    log_highlnP_weights, lin_highlnP_weights = gfx.plot_random_weights(sampler, nsteps, ndim, lin_weights, lin_err, cmddir, vvclim, log=False, svdir=svdir, truths=lin_truths, burn=burn)

    # do pg style plots using final 50th percentile weights:
    gfx.pgplot(obs, model, cmddir, bf, agemu, logz, av, dmod, vvclim, log_weights, filters, age_range, svdir=svdir)

    if vvcsig == 0.0:
        row_names = np.array(["t0", "age"])
    else:
        row_names = np.array(["t0", "t1", "t2", "t3", "t4", "t5", "t6","t7","t8","t9","age"])

    if agesig > 0.0:
        row_names = np.append(row_names, "age_sig")

    print(row_names)
    print(len(row_names))
    print(len(log_highlnP_weights))
    print(len(log_weights))
    print(len(log_err['higher']))
    print(len(log_err['lower']))
    print(len(truths))

    np.savetxt(os.path.join(cmddir, svdir, 'log_solutions.txt'),
                X=np.c_[row_names, log_highlnP_weights, log_weights, log_err['higher'], log_err['lower'], truths], delimiter='\t',fmt="%s", header="MAP\t 50th Percentile\t +err\t -err\t truth")
    np.savetxt(os.path.join(cmddir, svdir, 'lin_solutions.txt'),
                X=np.c_[row_names, lin_highlnP_weights, lin_weights, lin_err['higher'], lin_err['lower'], truths], delimiter='\t',fmt="%s", header="MAP\t 50th Percentile\t +err\t -err\t truth")

#=================================================================================================================================

    print("DONE")
