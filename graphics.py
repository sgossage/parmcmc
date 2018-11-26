import numpy as np
import matplotlib as mpl
mpl.use('Agg')

import matplotlib.pyplot as plt
import corner
import os
from matplotlib.ticker import FuncFormatter

from match.scripts import cmd
from match.MISTscripts.plotruns import *
from MIST_codes.scripts import read_mist_models as rmm
from mistmatch_filters import match_to_mist
import fileio as fio
from gausweight import *

def MyFormatter(x,lim):
    if x == 0:
        return 0
    if x < 0:
        x = abs(x)
        return '-{0}e{1}'.format(x/10**np.floor(np.log10(x)),int(np.log10(x)))
    elif x > 0:
        return '{0}e{1}'.format(x/10**np.floor(np.log10(x)),int(np.log10(x)))

# This does a pg style plot of the models and data, given a set of weights for the models:
def pgplot(obs, model, cmddir, bf, age, logz, vvc, av, dmod, weights, filters, svname=None, log=False, svdir=None):

    """
        Creates a MATCH pg style plot of data, model, data-model, and -2lnP map.
    """

    composite_cmd = cmd.CMD(fio.get_cmdf(cmddir, bf, age, logz, 0.0, av, dmod))#, ymag='I')
    composite_cmd.cmd['Nsim'] = np.zeros(len(composite_cmd.cmd['Nsim']))

    vvc_range = np.arange(0.0, 1.0, 0.1)
    Nrot = len(vvc_range)
    age_range = np.arange(8.5, 9.5, 0.02)
    mu = weights[Nrot]
    try:
        sigma = weights[Nrot+1]
    except IndexError:
        sigma = 0.0

    ageweights = gen_gaussweights(age_range, mu, sigma)
    #for i, avvc in enumerate(vvc_range):
    #    for j, anage in enumerate(age_range):

    #        a_cmd = cmd.CMD(fio.get_cmdf(cmddir, bf, anage, logz, avvc, av, dmod))#, ymag='I')
            # 1 * (jth age weight, added i times) * ith rotation rate.
    #        a_cmd.cmd['Nsim'] = (a_cmd.cmd['Nsim'] / np.sum(a_cmd.cmd['Nsim'])) * (best_gweights[j]) * (10**weights[i])

            # add each cmd (re-weighted by solutions) to the composite CMD model.
    #        composite_cmd.cmd['Nsim'] += a_cmd.cmd['Nsim']

    vvcweights = weights[:Nrot]    

    composite_cmd.cmd['Nsim'] += np.sum((10**vvcweights[:, np.newaxis])*np.sum(ageweights[:,np.newaxis]*model, axis=1), axis=0)


    composite_cmd.cmd['Nobs'] = obs

    # arbitrary vvc used -- just need a file name to break up.
    #fn_base = (fio.get_cmdf(cmddir, bf, age, logz, 0.0, av, dmod).split('/')[-1]).split('.out.cmd')[0]
    #fehval = float((fn_base.split('logz')[-1]).split('_')[0])
    #lageval = float((fn_base.split('_t')[-1]).split('_')[0])
    #avval = float((fn_base.split('_av')[-1]).split('_')[0])
    #dmodval = float((fn_base.split('_dmod')[-1]).split('_')[0])

    print(max(composite_cmd.cmd['Nsim']))

    filters, photstrs = match_to_mist(filters) 
    photstr = photstrs[0]
    print(filters)
    redmag_name = filters[1]
    bluemag_name = filters[0]

    color_name = "{:s}-{:s}".format(bluemag_name, redmag_name)

    iso00 = rmm.ISOCMD(round(float(logz), 2), 0.0, ebv= round(float(av), 2)/3.1, photstr=photstr, exttag='TP')
    iso00.set_isodata(round(float(mu), 2), color_name, bluemag_name, dmod=round(float(dmod), 2))

    iso06 = rmm.ISOCMD(round(float(logz), 2), 0.9, ebv= round(float(av), 2)/3.1, photstr=photstr, exttag='TP')
    iso06.set_isodata(round(float(mu), 2), color_name, bluemag_name, dmod=round(float(dmod), 2))

    # (x, y), i.e., (color, red mag) points of each isochrone in a list:
    mist_pts = [
                isoget_colmags(iso00, [color_name, bluemag_name], lage=round(float(mu), 2), dmod=round(float(dmod), 2)),
                isoget_colmags(iso06, [color_name, bluemag_name], lage=round(float(mu), 2), dmod=round(float(dmod), 2))
               ]

    # recalculate the d-m and signifigance hesses using the new hesses.
    composite_cmd.recalc()

    # create a MATCH pg style plot using the .cmd file:
    pgcmd_kwargs = {}
    pgcmd_kwargs['mist_pts'] = mist_pts
    if svname == None:
        if svdir == None:
            pgcmd_kwargs['figname'] = os.path.join(cmddir, 'match_pgplot.png')
        else:
            pgcmd_kwargs['figname'] = os.path.join(cmddir, svdir, 'match_pgplot.png')
    else:
        if svdir == None:
            pgcmd_kwargs['figname'] = os.path.join(cmddir, svname)
        else:
            pgcmd_kwargs['figname'] = os.path.join(cmddir, svdir, svname)


    # four panel plot:
    if log:
        pgcmd_kwargs['logcounts'] = True
        composite_cmd.pgcmd(**pgcmd_kwargs)
    else:
        composite_cmd.pgcmd(**pgcmd_kwargs)

    return

def chain_plot(nwalkers, ndim, chain, cmddir, lims, svdir=None, truths=None, lintruths=None, burn=-1000):

    """
        Plots our MCMC chains.
    """

    fig, axa = plt.subplots(ndim, 2, figsize=(10, 14), sharex=True)
    labels = [r"$\theta_0$", r"$\theta_1$", r"$\theta_2$", r"$\theta_3$", 
              r"$\theta_4$", r"$\theta_5$", r"$\theta_6$", r"$\theta_7$", 
              r"$\theta_8$", r"$\theta_9$", r"$\tau$", r"$\sigma_{\tau}$"]

    cmap = plt.cm.viridis

    #for i in range(nwalkers):
    for j in range(ndim):
            # only plot every 100th walker:
            #if not i % 10:
        logax = axa.T[:][0][j]
        linax = axa.T[:][1][j]
        steps = np.array([range(np.shape(chain)[1])]*nwalkers).flatten()
        logax.hist2d(steps, chain[:, :, j].flatten(), bins = 64, cmap = cmap)
        linax.hist2d(steps, (10**chain[:, :, j]).flatten(), bins = 64, cmap = cmap)
        #logax.plot(chain[i, :, j], alpha=0.4)
        #linax.plot(10**chain[i, :, j], alpha=0.4)
        logax.set_ylabel(labels[j])
        logax.axvline(x=np.shape(chain)[1]-1 + burn, c='r')
        linax.axvline(x=np.shape(chain)[1]-1 + burn, c='r')
        logax.axhline(y=truths[j], c='r', ls='--')
        linax.axhline(y=lintruths[j], c='r', ls='--')
            #else:
            #    break

    majorFormatter = FuncFormatter(MyFormatter)
    for ax in axa.T[:][1][:11]:
        ax.get_yaxis().set_major_formatter(majorFormatter)

    axa[-1][0].set_xlabel("step number")
    axa[-1][1].set_xlabel("step number")
    axa[0][0].set_title(r"Log $\theta_i$")
    axa[0][1].set_title(r"Linear $\theta_i$")

    # tight layout is bad here, find something else
    fig.tight_layout()

    if svdir == None:
        plt.savefig(os.path.join(cmddir, "chains_heatmap.png"))
    else:
        plt.savefig(os.path.join(cmddir, svdir, "chains_heatmap.png"))

    fig, axa = plt.subplots(ndim, 2, figsize=(10, 14), sharex=True)

    for i in range(nwalkers):
        for j in range(ndim):
            # only plot every 100th walker:
            #if not i % 10:
            logax = axa.T[:][0][j]
            linax = axa.T[:][1][j]
            #logax.hist2d(range(len(chain[i, :, j])), chain[i, :, j], bins = 50)
            #linax.hist2d(range(len(chain[i, :, j])), 10**chain[i, :, j], bins=50 )
            logax.plot(chain[i, :, j], alpha=0.4)
            linax.plot(10**chain[i, :, j], alpha=0.4)
            logax
            logax.set_ylabel(labels[j])
            logax.axvline(x=len(chain[i,:,j]) - len(chain[i,burn:,j]), c='k')
            linax.axvline(x=len(chain[i,:,j]) - len(chain[i,burn:,j]), c='k')
            logax.axhline(y=truths[j], c='r', ls='--')
            linax.axhline(y=lintruths[j], c='r', ls='--')
            #else:
            #    break

    # format ticks for all but the age std. deviation
    majorFormatter = FuncFormatter(MyFormatter)
    for ax in axa.T[:][1][:11]:
        ax.get_yaxis().set_major_formatter(majorFormatter)

    axa[-1][0].set_xlabel("step number")
    axa[-1][1].set_xlabel("step number")
    axa[0][0].set_title(r"Log $\theta_i$")
    axa[0][1].set_title(r"Linear $\theta_i$")

    # tight layout is bad here, find something else
    fig.tight_layout()

    if svdir == None:
        plt.savefig(os.path.join(cmddir, "chains.png"))
    else:
        plt.savefig(os.path.join(cmddir, svdir, "chains.png"))

    # cut out burn-in phase; using all but last 1000 steps as burn-in atm.
    samples = chain[:, burn:, :].reshape((-1, ndim))

    # corner plot of marginalized probability distributions for ea. parameter:
    if lintruths is None:
        fig = corner.corner(10**samples, labels=labels, quantiles=[0.16,0.5,0.84])
    else:
        fig = corner.corner(10**samples, labels=labels, quantiles=[0.16,0.5,0.84], truths=lintruths)

    if svdir == None:
        fig.savefig(os.path.join(cmddir, 'affine_invariant_triangle1_lin.png'))
    else:
        fig.savefig(os.path.join(cmddir, svdir, 'affine_invariant_triangle1_lin.png'))

    if truths is None:
        fig = corner.corner(samples, labels=labels, quantiles=[0.16,0.5,0.84])
    else:
        fig = corner.corner(samples, labels=labels, quantiles=[0.16,0.5,0.84], truths=truths)

    if svdir == None:
        fig.savefig(os.path.join(cmddir, 'affine_invariant_triangle1.png'))
    else:
        fig.savefig(os.path.join(cmddir, svdir, 'affine_invariant_triangle1.png'))

    return


def plot_random_weights(sampler, nsteps, ndim, weights, err, cmddir, log = True, svdir=None, truths=None, burn=-1000):

    """
        Creates a plot of our weights during the MCMC run, sampled from random walkers at 
    various iterations throughout the algorithm's execution.
    """

    vvcs = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    Nrot = len(vvcs)
    # figure for the weights of randomly drawn walkers at various iterations, along with 
    # 50th percentile weights w/ shaded region for 84th and 16th percentiles, and 
    # the highest lnP weights drawn as a black line as well.
    f,ax=plt.subplots(1,1,figsize=(16,9))

    # randomly pick a walker at some iteration from the chain and do pg plots:    
    highlnP_weights = np.array([-np.inf]*ndim)
    prevlnP = np.array([-1.0]*ndim)
    for i in range(abs(burn)):
        rand_weights = []
        for j in range(ndim):
            if not i % 200:
                rand_weights.append(np.random.choice(sampler.chain[:, burn+i, j]))

            # if max probabilitiy of set of walkers is greater than lasst greatest...
            if max(np.exp(sampler.lnprobability[:, burn+i])) > prevlnP[j]:
                highlnP_weights[j] = sampler.chain[:, burn+i, j][np.where(np.exp(sampler.lnprobability[:, burn+i]) == max(np.exp(sampler.lnprobability[:, burn+i])))[0][0]]
                prevlnP[j] = max(np.exp(sampler.lnprobability[:, burn+i]))

        # this can fail if rand_weights is not found.
        try:
            rand_weights = np.array(rand_weights)
            if log:
                # pg plot as this iteration, using weights from the randomly selected walkers:
                if svdir == None:
                    pgplot(cmddir, bf, age, logz, av, dmod, rand_weights, svname = "match_pgplot_{:d}.png".format(i))
                else:
                    pgplot(cmddir, bf, age, logz, av, dmod, rand_weights, svname = "{:s}/match_pgplot_{:d}.png".format(svdir, i))            

            else:
                # swap to linear weights:
                rand_weights = 10**rand_weights

            # plot the random walkers' weights:
            ax.plot(vvcs, rand_weights[:Nrot], c='b', alpha=0.3)

        except Exception as e:
            pass

    ax.set_xlabel(r'$\Omega/\Omega_c$', size=24)
    ax.set_ylabel('Weight', size=24)
    if log:
        ax.set_ylabel('log Weight', size=24)

    # plot the final 50th percentile weights as a red line; also use a filled region 
    # to indicate uncertainty:  
    ax.errorbar(vvcs, weights[:Nrot], yerr=[err['lower'][:Nrot], err['higher'][:Nrot]], c='r', ls=':')
    ax.errorbar(vvcs, weights[:Nrot] + err['higher'][:Nrot], c='r', ls='--')
    ax.errorbar(vvcs, weights[:Nrot] - err['lower'][:Nrot], c='r', ls='--')
    ax.fill_between(vvcs, weights[:Nrot] + err['higher'][:Nrot], y2=weights[:Nrot]-err['lower'][:Nrot], color='r', alpha=0.2)

    # plot the highest lnP weights as a black line as well on this figure:
    log_highlnP_weights = highlnP_weights
    lin_highlnP_weights = 10**highlnP_weights

    if truths is not None:
        ax.plot(vvcs, truths[:Nrot], c='k')

    if log:
        ax.plot(vvcs, log_highlnP_weights[:Nrot], c='c')
        if svdir == None:
            f.savefig(os.path.join(cmddir, 'log_soln_random.png'))
        else:
            f.savefig(os.path.join(cmddir, svdir, 'log_soln_random.png'))
    else:
        ax.plot(vvcs, lin_highlnP_weights[:Nrot], c='c')
        if svdir == None:
            f.savefig(os.path.join(cmddir, 'lin_soln_random.png'))
        else:
            f.savefig(os.path.join(cmddir, svdir, 'lin_soln_random.png'))

    return log_highlnP_weights, lin_highlnP_weights
