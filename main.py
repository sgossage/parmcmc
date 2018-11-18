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

import graphics as gfx
import prob as pr
from prob import lnprob
import fileio as fio

# right now 4 inputs available/required: cmddir (where .cmd files live), ncpu (num cpus for multiproc), 
# svdir (directory where plots are saved to), mode (either specifying mock or a run on observations).

# cmddir has format output/photbasename/mcmc/

if __name__ == "__main__":

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
    mode = sys.argv[4]
    vfilter = sys.argv[5]
    ifilter = sys.argv[6]
    filters = [vfilter, ifilter]    

    truths = {0.0: 1e-2, 0.1: 1e-2, 0.2: 1e-2, 0.3: 1e-2, 
              0.4: 1e-2, 0.5: 1e-2, 0.6: 1e-2}

    # generate specified model or get specified Hess:
    agemu = 9.00
    agesig = 0.3
    vvcmu = 0.3
    vvcsig = 0.2

    if mode == "mock":
        if vvc is not "gauss":
            if age is not "gauss":
                obscmd = cmd.CMD(fio.get_cmdf(cmddir, bf, age, logz, vvc, av, dmod))
                obs = obscmd.cmd['Nsim']
            else:
                obs = pr.genmod_agespread(cmddir, mass=1e6, agemu=agemu, agesig=agesig)

            truths[round(float(vvc), 1)] = np.sum(obs)
        else:
            if age is not "gauss":
                obs, truths = pr.genmod_vvcspread(cmddir, age, mass=1e6, vvcmu=vvcmu, vvcsig=vvcsig)        
            else:
                obs, truths = pr.genmod_agevvcspread(cmddir, mass=1e6, agemu=agemu, agesig=agesig, vvcmu=vvcmu, vvcsig=vvcsig)
    elif mode == "obs":
        obscmd = cmd.CMD(fio.get_cmdf(cmddir, bf, age, logz, vvc, av, dmod))
        obs = obscmd.cmd['Nobs']
        obsweight = np.sum(obs)
        truths = {x: obsweight for x in truths}

    # finalize format of truth values:
    lin_truths = np.array(list(truths.values()))
    truths = np.log10(lin_truths)
    if age is not "gauss":
        truths = np.append(truths, agemu)
        lin_truths = np.append(lin_truths, 10**agemu)
        ndim = 8    
    else:
        truths = np.append(truths, [agemu, agesig])
        lin_truths = np.append(lin_truths, [10**agemu, 10**agesig])
        ndim = 9

    # form the array of models spanning age and v/vc grid space:
    # structure is an NxM dimensional matrix of Hess diagrams, indexed 
    # by i for the ith vector along the rotation rate axis, and by j along 
    # the jth age axis. Each Hess diagram is normalized to sum to one.
    vvc_range = np.arange(0.0, 0.7, 0.1)
    age_range = np.arange(8.5, 9.5, 0.02)
    vvc_pts = []
    for i, a_vvc in enumerate(vvc_range):
        # step through in age, adding an age vector at each point in v/vc space
        age_vector = [] 
        for j, an_age in enumerate(age_range):

            a_cmd = cmd.CMD(fio.get_cmdf(cmddir, bf, an_age, logz, a_vvc, av, dmod))
            model_hess = a_cmd.cmd['Nsim']
            model_hess /= np.sum(model_hess)

            age_vector.append(model_hess)

        vvc_pts.append(np.array(age_vector))

    model = np.array(vvc_pts)
   
    # 9 dimensions (7 rotation rates, age, age std. deviation), 
    # however many walkers, number of iterations:
    nwalkers, nsteps = 256, 512 # 600, 2000
    burn = -128

    #ndim, nwalkers, nsteps = 9, 20, 200
    #burn = -100

    # walkers initialized via uniform distribution centered on data mass, w/ +/- 1 range.
    pos = []
    for i in range(nwalkers):
        posi = truths[:7] + np.random.uniform(-2, 2, 7)#np.random.uniform(-(np.log10(np.sum(obs))+0.01), 0.01, 7)
        posi = np.append(posi, 9.0 + np.random.uniform(-0.4, 0.4, 1))
        if ndim == 9:
            posi = np.append(posi, 0.1 + np.random.uniform(-0.05, 0.05, 1))

        pos.append(posi)

    lims = np.array([[truth - 2, truth + 2] for truth in truths[:7]]) # was log10(obs) +/- 2 dex

    print(truths)
    print(lin_truths)
    print(lims)

    # the affine-invariant emcee sampler:
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, kwargs = {"obs":obs, "model":model, "lims":lims}, threads=ncpu)

    print("Running MCMC...")
    # run mcmc:
    print("Thinning chains...")
    new_pos, sampler = burn_emcee(sampler, pos, [16, 32, 64, 128, 256])#, 128, 256, 512, 1024])
    print("Doing full run...")
    pos, prob, state = sampler.run_mcmc(new_pos, nsteps)

    print("Plotting results...")
    gfx.chain_plot(nwalkers, ndim, sampler.chain, cmddir = cmddir, lims=lims, svdir=svdir, truths=truths, lintruths=lin_truths, burn=burn)

    # plot ln P of full model:
    fig, ax = plt.subplots(1, figsize=(10,7))
    for i in range(nwalkers):
        ax.plot(sampler.lnprobability[i, burn:])

    ax.set_xlabel("Step Number")
    ax.set_ylabel("ln P")

    plt.tight_layout()

    plt.savefig(os.path.join(cmddir, svdir, 'chains_lnP.png'))

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
    ax.errorbar(vvc_range, log_weights[:7], yerr=[log_err['lower'][:7], log_err['higher'][:7]], c='r', ls='--')
    ax.plot(vvc_range, truths[:7], c='k')
    ax.set_xlabel(r'$\Omega/\Omega_c$', size=24)
    ax.set_ylabel('log Best Weight', size=24)
    f.savefig(os.path.join(cmddir, svdir, 'soln_vs_vvc.png'))

    f,ax=plt.subplots(1,1,figsize=(16,9))
    ax.errorbar(vvc_range, lin_weights[:7], yerr=[lin_err['lower'][:7], lin_err['higher'][:7]], c='r', ls='--')
    ax.plot(vvc_range, lin_truths[:7], c='k')
    ax.set_xlabel(r'$\Omega/\Omega_c$', size=24)
    ax.set_ylabel('Best Weight', size=24)
    f.savefig(os.path.join(cmddir, svdir, 'soln_vs_vvc_lin.png'))

    # do pg style plots using final 50th percentile weights:
    gfx.pgplot(obs, model, cmddir, bf, age, logz, vvc, av, dmod, log_weights, filters, svdir=svdir)
    _ = gfx.plot_random_weights(sampler, nsteps, ndim, log_weights, log_err, cmddir, log=True, svdir=svdir, truths=truths, burn=burn)
    log_highlnP_weights, lin_highlnP_weights = gfx.plot_random_weights(sampler, nsteps, ndim, lin_weights, lin_err, cmddir, log=False, svdir=svdir, truths=lin_truths, burn=burn)

    row_names = np.array(["t0", "t1", "t2", "t3", "t4", "t5", "t6","age"])
    if ndim == 9:
        row_names = np.append(row_names, "age_sig")
    np.savetxt(os.path.join(cmddir, svdir, 'log_solutions.txt'),
                X=np.c_[row_names, log_highlnP_weights, log_weights, log_err['higher'], log_err['lower']], delimiter='\t',fmt="%s", header="MAP\t50th Percentile\t+err\t-err")
    np.savetxt(os.path.join(cmddir, svdir, 'lin_solutions.txt'),
                X=np.c_[row_names, lin_highlnP_weights, lin_weights, lin_err['higher'], lin_err['lower']], delimiter='\t',fmt="%s", header="MAP\t50th Percentile\t+err\t-err")

    print("DONE")
