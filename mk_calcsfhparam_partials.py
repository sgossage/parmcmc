#!/usr/bin/env python
import matplotlib as mpl
mpl.use('Agg')
from match.scripts.fileio import calcsfh_input_parameter

import os
import sys
import numpy as np
from mk_paramfs import mk_agegrid

# sys.argv[x]: 1, 'v' filter (bluer); 2, 'i' filter (redder); 3, photometry file; 4

def parse_fname(photfn):

    # photometry file has bfx.x_avx.x_tx.xx_x.xx_logZx.xx_dmod_x.xx.phot.mags
    # this takes these parameters from the file name so that they may be used 
    # in calcsfh parameter file specification.

    # Add these parameters to the cluster photometry file names too.

    bf = float((photfn.split('bf')[-1]).split('_')[0])
    av = float((photfn.split('av')[-1]).split('_')[0])
    agebin = (photfn.split('_t')[-1]).split('_')[:2]
    try:
        agebin = list(map(float, agebin))
    except ValueError:
        # age string is just a delta tx.xx, no subsequent endpoint.
        agebin = [float((photfn.split('_t')[-1]).split('_')[0])]
        # use smallest model resolution, 0.02 to define endpoint.
        agebin.append(agebin[0]+0.02)
    logZ = float((photfn.split('logZ')[-1]).split('_')[0])
    if "dmod" in photfn:
        dmod = float((photfn.split('dmod')[-1]).split('_')[0])
    else:
        # assumes that dmod is 0 if not included in file name.
        dmod = 0.0

    return bf, av, agebin, logZ, dmod

if __name__ == '__main__':

    """
        Parameters are specified by the photometry file name. Right now,
    the search in av and dmod (and bf) are disabled. The magnitude limit 
    is set to 8, this may need to be changed per cluster/data. Exclude & 
    combine gates need to be changed manually right now. Ways to automate?

    Also, makes an "age grid" at the end. This makes separate calcsfh 
    parameter files from the main file, where each new file varies the 
    age bin. This is done in order to ultimately create an age grid of 
    Hess diagrams from calcsfh, each fit in ssp mode, in order to act 
    as potential components of a composite model population.
    """

    # Set the parameters...specific to a given cluster?
    vfilter_name = sys.argv[1]
    ifilter_name = sys.argv[2]
    photfn = sys.argv[3]

    # not dynamic. make it dynamic??
    phot_dir = 'phot_mock'

    v = np.genfromtxt(os.path.join(os.getcwd(), phot_dir, photfn), usecols=(0,))
    i = np.genfromtxt(os.path.join(os.getcwd(), phot_dir, photfn), usecols=(1,))

    # try getting errors:
    try:
        verr = np.genfromtxt(os.path.join(os.getcwd(), phot_dir, photfn.replace('.phot', '.err')), usecols=(0,))
        berr = np.genfromtxt(os.path.join(os.getcwd(), phot_dir, photfn.replace('.phot', '.err')), usecols=(1,))
        bverr = np.genfromtxt(os.path.join(os.getcwd(), phot_dir, photfn.replace('.phot', '.err')), usecols=(2,))
    except IOError:
        pass 

    bf, av, agebin, logZ, dmod = parse_fname(photfn)

    # mag cap is not dynamic -- can be troublesome.
    vmi = v - i
    #mag_cap = 8.0
    #vmax = mag_cap
    #imax = mag_cap
    vmax = np.amax(v) + 1.5
    imax = np.amax(i) + 1.5

    vmin = np.amin(v) - 1.5
    imin = np.amin(i) - 1.5
    vmi_max = np.amax(vmi) + 0.5
    vmi_min = np.amin(vmi) - 0.5

    # dav needs to be dynamic if not fixed.
    dav = 0.0
    av0 = av - dav*2.0
    if av0 < 0.0:
        av0 = 0.0
    av1 = av + dav*2.0

    # dmod needs to be dynamic if not fixed.
    ddmod = 0.0
    dmod0 = dmod - ddmod*2.0
    if dmod0 < 0.0:
        dmod0 = 0.0
    dmod1 = dmod + ddmod*2.0

    # set the magnitude and color bin sizes to avg. error size; use suggested dm=0.10 and dc=0.05 
    # as lower lims to bin size.
    try:
        # all zeros used to indicate column not available -- maybe change to e.g. inf??
        if all(verr == 0):
            raise NameError
        vbin = float("{:.2f}".format(np.mean(verr)))
        if vbin < 0.10:
            vbin = 0.10
    
    except NameError:
        vbin = 0.10
    
    try:
        if all(bverr == 0):
            raise NameError
        vibin = float("{:.2f}".format(np.mean(bverr)))
        if vibin < 0.05:
            vibin = 0.05

    except NameError:
        vibin = 0.05

    # params set here:
    params = {'dmod0': dmod0, 'dmod1': dmod1, 'ddmod': ddmod, 'av0': av0, 'av1': av1, 'dav': dav, 
              'dlogz': 0.02, 'logzmax': logZ+0.01, 'logzmin': logZ-0.01, 'tmax': max(agebin), 'tmin': min(agebin), 'tbin': 0.02, 
              'v': vfilter_name, 'i': ifilter_name, 'vmin': vmin, 'vmax': vmax, 'imin': imin, 'imax':imax, 
              'vimin': vmi_min, 'vimax': vmi_max, 'vistep': vibin, 'vstep': vbin, 'bf': bf}

    # Exclude & combine gates (should be dynamic for each cluster):
    nexclude_gates = 0
    ex_gate_pts = None #[1.0, 6.0, vmi_max, 6.0, vmi_max, vmin, 1.0, vmin]
    if ex_gate_pts != None:
        exclude_gates = "{:d} ".format(nexclude_gates) + " ".join([str(element) for element in ex_gate_pts])
    else:
        exclude_gates = "{:d}".format(nexclude_gates)

    ncombine_gates = 0
    cb_gate_pts = None
    if cb_gate_pts != None:
        combine_gates = "{:d} ".format(ncombine_gates) + " ".join([str(element) for element in cb_gate_pts])
    else:
        combine_gates = "{:d}".format(ncombine_gates)
    
    gate_line = exclude_gates + " " + combine_gates + "\n"

    # Write to file:
    photbase = photfn.split(".phot")[0]
    paramf_name = os.path.join(os.getcwd(), 'csfh_param',  "{:s}.param".format(photbase))

    with open(paramf_name, 'w+') as outf:
        # Use Phil & Dan's code to auto write param file w/ above parameters (doesn't do exclude gates automatically): 
        outf.write(calcsfh_input_parameter(power_law_imf=False, **params))
      
        # Get lines of that file:
        outf.seek(0)
        lines = outf.readlines()
        outf.seek(0)
        # Manually replace the exclude gates line:
        lines[7] = gate_line
        for line in lines:
            outf.write(line)

    # creates copies of the calcsfh param file where age is varied.
    mk_agegrid(paramf_name, t0=8.30, tf=9.80, tbin=0.02)
