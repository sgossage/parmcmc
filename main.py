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

    photbase = sys.argv[1].split('/')[1]
    bf, av, agebin, logz, dmod = fio.parse_fname(photbase, mode="str")
    print(bf, av, agebin, logz, dmod)
    print(type(bf), type(av), type(agebin[0]), type(agebin[1]), type(logz), type(dmod))
    
    # these should be made into dynamix inputs. 0.00, "gauss", "-0.30", "gauss", "0.0", "0.00", "Tycho_B", "Tycho_V"
    #bf, age, logz, vvc, av, dmod, vfilter, ifilter = "0.25", "gauss", "-0.30", "gauss", "0.16", "18.37", "UVIS475W", "UVIS814W"
    age, vvc = "gauss", "gauss"
    cmddir = sys.argv[1]
    # number of cpus, for multiprocessing:
    ncpu = int(sys.argv[2])

    svdir = sys.argv[3]
    # if mode is "mock", a synthetic cluster will be created and used as data (for testing)
    # if mode is "obs", a Hess diagram will be made from list of observed magnitudes
    mode = sys.argv[4]

    # MATCH style filter names:
    vfilter = sys.argv[5]
    ifilter = sys.argv[6]
    filters = [vfilter, ifilter]

    # Controls for age and v/vc distributions (agesig = 0.0 means no age spread/vvcsig = 0.0 means no v/vc spread).
    agemu = round(float(sys.argv[7]), 2)
    agesig = round(float(sys.argv[8]), 2)
    vvcmu = round(float(sys.argv[9]), 1)
    vvcsig = round(float(sys.argv[10]), 1)

    #truths = {0.0: 1e-2, 0.1: 1e-2, 0.2: 1e-2, 0.3: 1e-2, 
    #          0.4: 1e-2, 0.5: 1e-2, 0.6: 1e-2, 0.7: 1e-2,
    #          0.8: 1e-2, 0.9: 1e-2}

#=================================================================================================================================
# Model generation for mock data, or else assigning "observations" to the given observed data.
# Truths are set here as well.

    # generate specified model or get specified Hess:
    #agemu = 9.20  # 9.00 for NGC2249 (and for mocks), 9.20 for NGC 2203
    #agesig = 0.0  # 0.3 was the selection for NGC 2203
    #vvcmu = 0.3   # default 0.3 as the mean
    #vvcsig = 0.2  # 0.2 default std. deviation for rotation distribution

    if vvcsig > 0.0:
        Nrot = 10     # 10 is the current max, will create models @ 0.0, 0.1, ..., 0.9; set to 1 for no rotation, v/vc = 0.0
    else:
        Nrot = 1

    vvclim = round(Nrot / 10.0, 1)
    vvc_range = np.arange(0.0, vvclim, 0.1)
    age_range = np.arange(8.5, 9.5, 0.02)
    # default, dummy truths:
    truths = {rot:1e-2 for rot in vvc_range}
    mass = 5E4 # mock cluster "mass" or total counts; 1e6 was default

    if mode == "mock":
        if vvc is not "gauss":
            if age is not "gauss" or agesig == 0.0:
                obscmd = cmd.CMD(fio.get_cmdf(cmddir, bf, age, logz, vvc, av, dmod))
                obs = obscmd.cmd['Nsim']
            else:
                obs = pr.genmod_agespread(cmddir, mass=maass, agemu=agemu, agesig=agesig, vvclim=vvclim)

            truths[round(float(vvc), 1)] = np.sum(obs)
        else:
            if age is not "gauss" or agesig == 0.0:
                obs, truths = pr.genmod_vvcspread(cmddir, age, mass=mass, vvcmu=vvcmu, vvcsig=vvcsig, vvclim=vvclim)        
            else:
                obs, truths = pr.genmod_agevvcspread(cmddir, mass=mass, agemu=agemu, agesig=agesig, vvcmu=vvcmu, vvcsig=vvcsig, vvclim=vvclim)

        obsweight = mass

    # or else just use the observed Hess diagram values; don't create a model.
    # truth is "unknown" here for the vvc weights; each is set to the total observed counts.
    elif mode == "obs":
        obscmd = cmd.CMD(fio.get_cmdf(cmddir, bf, age, logz, vvc, av, dmod))
        obs = obscmd.cmd['Nobs']
        obsweight = np.sum(obs)
        truths = {x: obsweight for x in truths}

    # finalize format of truth values:
    lin_truths = np.array(list(truths.values()))
    truths = np.log10(lin_truths)
    if age is not "gauss" or agesig == 0.0:
        truths = np.append(truths, agemu)
        lin_truths = np.append(lin_truths, 10**agemu)
        ndim = Nrot+1    
    else:
        truths = np.append(truths, [agemu, agesig])
        lin_truths = np.append(lin_truths, [10**agemu, 10**agesig])
        ndim = Nrot + 2

#=================================================================================================================================
# Build grid of models from which emcee will draw to do the data-model fit.

    model = fio.gather_models(cmddir, bf, age_range, logz, vvc_range, av, dmod)

#=================================================================================================================================
# Perform MCMC algorithm to attain best-fit parameters.

    # 9 dimensions (7 rotation rates, age, age std. deviation), 
    # however many walkers, number of iterations:
    nwalkers, nsteps = 1024, 512 # 4096 # 600, 2000
    burn = -int(nsteps/2.) #4.)

    #ndim, nwalkers, nsteps = 9, 20, 200
    #burn = -100

    # walkers initialized via uniform distribution centered on data mass, w/ +/- 1 range.
    # Need a better way to initialize walkers...try setting them up on hyperplane of solutions.
    #age_range = np.arange(8.5,9.5,0.02)
    #age_posi = np.array([])
    #agesig_posi = np.array([])
    #rotw = np.array([])
    #y = obs
    #for i in range(nwalkers):
    #    age_posi = np.append(age_posi, 9.0 + np.random.uniform(-0.2, 0.2, 1))
        
    #    if ndim == Nrot+2:
    #        agesig_posi = np.append(agesig_posi, 0.1 + np.random.uniform(-0.05, 0.05, 1))
    #        ageweights = gen_gaussweights(age_range, age_posi[i], agesig_posi[i])
    #    else:
    #        ageweights = gen_gaussweights(age_range, age_posi[i], sigma)

    #    beta = np.array([-1.]*7)
    #    #print(y.shape)
    #    while not all(beta > 0.0):
    #        X = np.sum(ageweights[:,np.newaxis]*model, axis=1)
            #print(X.shape)
    #        X += np.random.uniform(0,0.25,X.shape)
    #        X = X.T
            #print(X.shape)
    #        XTXinv = np.linalg.inv(np.matmul(X.T, X))
            #print(XTXinv.shape)
    #        XTXinvXT = np.matmul(XTXinv, X.T)
            #print(XTXinvXT.T.shape)
    #        XTXinvXTy = np.matmul(XTXinvXT, y)
    #        beta = XTXinvXTy
    #    beta = np.log10(beta)
    #    if i == 0:
    #        rotw = beta
    #    else:
    #        rotw = np.vstack((rotw, beta))
    #print(beta.shape)
    #print(rotw.shape)
    #print(model.shape)
    #print(X.shape)
    #print(beta)
    #print(all(beta > 0.0))
    #print(np.log10(beta))
    #print(beta.shape)
    #print(np.sum(X*beta))
    #print(np.sum(y))
    #sys.exit()

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
        #posi = truths[:Nrot] + np.random.uniform(-1, 1, Nrot)#np.random.uniform(-(np.log10(np.sum(obs))+0.01), 0.01, 7)
        #posi = np.append(posi, age_posi[i])#
        # initial age positions randomized according to uniform dist. about supplied mean age.
        posi = np.append(posi, agemu + np.random.uniform(-0.2, 0.2, 1))
        if ndim == Nrot+2:
            #posi = np.append(posi, agesig_posi[i])
            # initial age gaussian std. dev. randomized according to uniform dist. about supplied sigma.
            posi = np.append(posi, agesig + np.random.uniform(-0.05, 0.05, 1))

        pos.append(posi)
    print(pos[0])
    print("Sample sum of v/vc weights (log10) = ", np.log10(np.sum(10**pos[0][:Nrot])))

    # debug plot
    #with open('positions.txt', 'w') as f:
    #    print(pos[:][:-1], file=f)

    # prior distribution limits for v/vc weights (assuming flat priors).
    lims = np.array([[max([0.0, np.log10(obsweight*1e-4)]), np.log10(obsweight*1.1)] for truth in truths[:Nrot]]) # was log10(obs) +/- 2 dex

    print(truths)
    print(lin_truths)
    print(lims)
    print(len(lims))
    print(ndim)
    #sys.exit()

    # the affine-invariant emcee sampler:
    print("Setting up MCMC sampler...")
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, kwargs = {"obs":obs, "model":model, "lims":lims}, threads=ncpu)
    #sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_part, pool=pool)

    print("Running MCMC...")
    # run mcmc:
    print("Thinning chains...")
    new_pos, sampler = burn_emcee(sampler, pos, [16, 32, 64, 128, 256, 512, 1024])#16, 32, 64, 128, 256, 512, 1024])
    print("Doing full run...")
    pos, prob, state = sampler.run_mcmc(new_pos, nsteps)
#    pool.close()

#=================================================================================================================================
# Plot the results.

    print("Plotting results...")
    gfx.chain_plot(nwalkers, ndim, sampler, cmddir = cmddir, vvclim=vvclim, svdir=svdir, truths=truths, lintruths=lin_truths, burn=burn)

    # plot ln P of full model:
    #fig, ax = plt.subplots(1, figsize=(10,7))
    #for i in range(nwalkers):
    #    ax.plot(sampler.lnprobability[i, burn:])

    #ax.set_xlabel("Step Number")
    #ax.set_ylabel("ln P")

    #plt.tight_layout()

    #plt.savefig(os.path.join(cmddir, svdir, 'chains_lnP.png'))

    # plot ln P of full model:
    #fig, ax = plt.subplots(1, figsize=(10,7))
    #ax.hist2d(range(len(chain[i, :, j])), chain[i, :, j], bins = 30, cmap = cmap)
    #for i in range(nwalkers):
    #    ax.plot(sampler.lnprobability[i, burn:])

    #ax.set_xlabel("Step Number")
    #ax.set_ylabel("ln P")

    #plt.tight_layout()

    #plt.savefig(os.path.join(cmddir, svdir, 'chains_lnP.png'))

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

    #vvcs = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

    # plot of best solutions (weights) vs. v/vc:
    f,ax=plt.subplots(1,1,figsize=(16,9))
    
    print(log_weights[:Nrot])
    print(log_err['lower'][:Nrot])
    print(log_err['higher'][:Nrot])

    ax.errorbar(vvc_range, log_weights[:Nrot], yerr=[log_err['lower'][:Nrot], log_err['higher'][:Nrot]], c='r', ls='--')
    ax.plot(vvc_range, truths[:Nrot], c='k')
    ax.set_xlabel(r'$\Omega/\Omega_c$', size=24)
    ax.set_ylabel('log Best Weight', size=24)
    f.savefig(os.path.join(cmddir, svdir, 'soln_vs_vvc.png'))

    f,ax=plt.subplots(1,1,figsize=(16,9))
    ax.errorbar(vvc_range, lin_weights[:Nrot], yerr=[lin_err['lower'][:Nrot], lin_err['higher'][:Nrot]], c='r', ls='--')
    ax.plot(vvc_range, lin_truths[:Nrot], c='k')
    ax.set_xlabel(r'$\Omega/\Omega_c$', size=24)
    ax.set_ylabel('Best Weight', size=24)
    f.savefig(os.path.join(cmddir, svdir, 'soln_vs_vvc_lin.png'))

    _ = gfx.plot_random_weights(sampler, nsteps, ndim, log_weights, log_err, cmddir, vvclim, log=True, svdir=svdir, truths=truths, burn=burn)
    log_highlnP_weights, lin_highlnP_weights = gfx.plot_random_weights(sampler, nsteps, ndim, lin_weights, lin_err, cmddir, vvclim, log=False, svdir=svdir, truths=lin_truths, burn=burn)

    # do pg style plots using final 50th percentile weights:
    gfx.pgplot(obs, model, cmddir, bf, age, logz, av, dmod, vvclim, log_weights, filters, svdir=svdir)

    row_names = np.array(["t0", "t1", "t2", "t3", "t4", "t5", "t6","t7","t8","t9","age"])
    if ndim == Nrot+2:
        row_names = np.append(row_names, "age_sig")
    elif ndim == 3:
        row_names = np.array(["t0", "age", "age_sig"])
    np.savetxt(os.path.join(cmddir, svdir, 'log_solutions.txt'),
                X=np.c_[row_names, log_highlnP_weights, log_weights, log_err['higher'], log_err['lower']], delimiter='\t',fmt="%s", header="MAP\t50th Percentile\t+err\t-err")
    np.savetxt(os.path.join(cmddir, svdir, 'lin_solutions.txt'),
                X=np.c_[row_names, lin_highlnP_weights, lin_weights, lin_err['higher'], lin_err['lower']], delimiter='\t',fmt="%s", header="MAP\t50th Percentile\t+err\t-err")

#=================================================================================================================================

    print("DONE")
