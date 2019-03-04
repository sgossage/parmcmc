import glob
import os
import numpy as np
from match.scripts import cmd

def gather_models(cmddir, bf, age_range, logz, vvc_range, av, dmod):
    # form the array of models spanning age and v/vc grid space:
    # structure is an NxM dimensional matrix of Hess diagrams, indexed 
    # by i for the ith vector along the rotation rate axis, and by j along 
    # the jth age axis. Each Hess diagram is normalized to sum to one.
    vvc_pts = []
    for i, a_vvc in enumerate(vvc_range):
        # step through in age, adding an age vector at each point in v/vc space
        age_vector = [] 
        for j, an_age in enumerate(age_range):

            a_cmd = cmd.CMD(get_cmdf(cmddir, bf, an_age, logz, a_vvc, av, dmod))
            model_hess = a_cmd.cmd['Nsim']
            model_hess /= np.sum(model_hess)

            age_vector.append(model_hess)

        vvc_pts.append(np.array(age_vector))

    gates = a_cmd.cmd['gate']
    ugates = np.unique(gates)
    if len(ugates) > 1:
        dinds = np.digitize(gates, bins=ugates, right=True)
    else:
        dinds = None

    model = np.array(vvc_pts)

    return model, dinds

def parse_fname(photfn, mode="float"):

    # used by mk_calcsfh_param.py to make a calcsfh paramfile

    # photometry file has bfx.x_avx.x_tx.xx_x.xx_logZx.xx_dmod_x.xx.phot.mags
    # this takes these parameters from the file name so that they may be used 
    # in calcsfh parameter file specification.

    # Add these parameters to the cluster photometry file names too.

    bf = (photfn.split('bf')[-1]).split('_')[0]
    av = (photfn.split('av')[-1]).split('_')[0]
#    agebin = (photfn.split('_t')[-1]).split('_')[:2]
#    try:
#        agebin = list(map(float, agebin))
#    except ValueError:
        # age string is just a delta tx.xx, no subsequent endpoint.
    agebin = [float((photfn.split('_t')[-1]).split('_')[0])]
    # use smallest model resolution, 0.02 to define endpoint.
    agebin.append(agebin[0]+0.02)
    agebin = ["{:.2f}".format(agebin[0]), "{:.2f}".format(agebin[1])]

    logZ = (photfn.split('logZ')[-1]).split('_')[0]
    if "dmod" in photfn:
        dmod = (photfn.split('dmod')[-1]).split('_')[0]
    else:
        # assumes that dmod is 0 if not included in file name.
        dmod = "0.00"

    if mode == "float":
        return float(bf), float(av), list(map(float, agebin)), float(logZ), float(dmod)

    elif mode == "str":
        return bf, av, agebin, logZ, dmod

def get_cmdf(cmddir, bf, age, logz, vvc, av, dmod):

    """
        Returns the .out.cmd file of a given set of parameters from a calcsfh run. These files 
    contain the MATCH constructed Hess diagrams.
    """

    vvc = str(vvc)
    #if vvc == "gauss":
        # arbitrary assignment [0.0, 0.6]
    #    vvc = "0.3"
    #age = str(age)
    #if age == "gauss":
        # arbitrary assignment [8.5,9.5]
    #    age = 9.00
    # for the path to the .cmd file:
    cmdpath = os.path.join(cmddir,
                                   'bf{:s}_t{:.2f}_logz{:s}_vvc{:s}_av{:s}_dmod{:s}.out.cmd'.format(bf, age,
                                                                                                     logz, vvc,
                                                                                                     av, dmod)
                                                                                                     )

    print("Looking for {:s}...".format(cmdpath))
    cmdfn = glob.glob(cmdpath)[0]

    return cmdfn
