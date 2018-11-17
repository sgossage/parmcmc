import glob
import os

def parse_fname(photfn, mode="float"):

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
    if vvc == "gauss":
        # arbitrary assignment [0.0, 0.6]
        vvc = "0.3"
    #age = str(age)
    if age == "gauss":
        # arbitrary assignment [8.5,9.5]
        age = 9.00
    # for the path to the .cmd file:
    cmdpath = os.path.join(cmddir,
                                   'bf{:s}_t{:.2f}_logz{:s}_vvc{:s}_av{:s}_dmod{:s}.out.cmd'.format(bf, age,
                                                                                                     logz, vvc,
                                                                                                     av, dmod)
                                                                                                     )

    #print("Looking for {:s}...".format(cmdpath))
    cmdfn = glob.glob(cmdpath)[0]

    return cmdfn
