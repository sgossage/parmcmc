from match.scripts import cmd
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from mistmatch_filters import match_to_mist
import fileio as fio
from gausweight import *
import os
import sys

if __name__ == '__main__':

    plot_i = sys.argv[1]

    f = plt.figure(figsize=(40,20))#, axa = plt.subplots(5,5, figsize=(20,20))
    gs = gridspec.GridSpec(5, 10)
    axa = []
    for i in range(5):
        axcols = []
        for j in range(5):
    	    axcols.append(f.add_subplot(gs[i, j]))
        axa.append(np.array(axcols))
    axa = np.array(axa)

    print(axa[0,0])
    #ax0 = plt.subplot(gs[1:-1, 1:-1])
    #ax1 = plt.subplot(gs[0, 0])
    #ax2 = plt.subplot(gs[0, 1])
    #ax3 = plt.subplot(gs[0, 2])
    #ax4 = plt.subplot(gs[0, 3])
    #ax5 = plt.subplot(gs[0, 4])

    #ax6 = plt.subplot(gs[1, 0])
    #ax7 = plt.subplot(gs[1, -1])
    #ax8 = plt.subplot(gs[2, 0])
    #ax9 = plt.subplot(gs[2, -1])

    #ax10 = plt.subplot(gs[-1, 0])
    #ax11 = plt.subplot(gs[-1, 1])
    #ax12 = plt.subplot(gs[-1, 2])
    #ax13 = plt.subplot(gs[-1, 3])
    #ax14 = plt.subplot(gs[-1, 4])
    #axa = [ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9,ax10,ax11,ax12,ax13,ax14]
    bf, age, logz, av, dmod = "0.00", 9.00, '-0.30', '0.0', '0.00'
    cmddir = 'output/bf0.00_av0.0_SFR0.01_t9.00_9.02_logZ-0.30_vvc0.4_Tycho_BTycho_V_ex/'

    # empty out a hess to use as a composite model later:
    composite_cmd = cmd.CMD(fio.get_cmdf(cmddir, bf, age, logz, 0.0, av, dmod))#, ymag='I')
    composite_cmd.cmd['Nsim'] = np.zeros(len(composite_cmd.cmd['Nsim']))
    obsweight = np.sum(composite_cmd.cmd['Nobs'])    

    # need random weights and mu, sigma
    filters = ['Tycho_B', 'Tycho_V']

    vvc_range = np.arange(0.0, 1.0, 0.2)
    Nrot = len(vvc_range)
    age_range = np.arange(8.5, 9.5, 0.2)
    mu = np.random.uniform(8.5, 9.3)
    sigma = 10**np.random.uniform(-2, 0)
#    try:
#        sigma = weights[Nrot+1]
#    except IndexError:
#        sigma = 0.0

    ageweights = gen_gaussweights(age_range, mu, sigma)
    rand_weights = np.random.uniform(0, np.log10(obsweight), len(vvc_range))

    cmddir = 'output/bf0.00_av0.0_SFR0.01_t9.00_9.02_logZ-0.30_vvc0.4_Tycho_BTycho_V_ex/'
    # step through each hess, and reweigh it
    vmin = 0
    vmax = np.max(rand_weights)
    for i, avvc in enumerate(vvc_range):
        for j, anage in enumerate(age_range):

            a_cmd = cmd.CMD(fio.get_cmdf(cmddir, '0.00', anage, '-0.30', avvc, '0.0', '0.00'))#, ymag='I')
            # 1 * (jth age weight, added i times) * ith rotation rate.
            a_cmd.cmd['Nsim'] = (a_cmd.cmd['Nsim'] / np.sum(a_cmd.cmd['Nsim'])) * (ageweights[j]) * (10**rand_weights[i])

            # plot on axis grid
            #if i != len(vvc_range)-1:
            a_cmd.plthess(ax = axa[i,j], figname='test_plot.png', hess_i = 1, save=False, vmin=vmin, vmax=vmax)
                #break
            #    pass
            #else:
                #a_cmd.plthess(ax = axa[i,j], figname='test_plot.png', hess_i = 1, save=True)
                #break
            #    pass
            #axa[i,j].plot([0,1],[0,1])


            # add each cmd (re-weighted by solutions) to the composite CMD model.
            composite_cmd.cmd['Nsim'] += a_cmd.cmd['Nsim']
        #break

    composite_cmd.plthess(ax = f.add_subplot(gs[:, 5:]), figname='test_plot.png', hess_i = 1, save=False, vmin=None, vmax=None)
    plt.tight_layout()
    f.savefig('test_plot_{:s}.png'.format(plot_i))
    #vvcweights = weights[:Nrot]    

    #composite_cmd.cmd['Nsim'] += np.sum((10**vvcweights[:, np.newaxis])*np.sum(ageweights[:,np.newaxis]*model, axis=1), axis=0)

    #composite_cmd.cmd['Nobs'] = obs

    print(max(composite_cmd.cmd['Nsim']))

    filters, photstrs = match_to_mist(filters) 
    photstr = photstrs[0]
    print(filters)
    redmag_name = filters[1]
    bluemag_name = filters[0]

    color_name = "{:s}-{:s}".format(bluemag_name, redmag_name)

    # recalculate the d-m and signifigance hesses using the new hesses.
    composite_cmd.recalc()

    # create a MATCH pg style plot using the .cmd file:
    pgcmd_kwargs = {}
    svdir = None
    if svdir == None:
        pgcmd_kwargs['figname'] = os.path.join('test_pgplot.png')
    else:
        pgcmd_kwargs['figname'] = os.path.join(svdir, 'test_pgplot.png')


    # four panel plot:
    #if log:
    #    pgcmd_kwargs['logcounts'] = True
    #    composite_cmd.pgcmd(**pgcmd_kwargs)
    #else:
    composite_cmd.pgcmd(**pgcmd_kwargs)
	
