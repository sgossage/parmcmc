import glob
import os

def get_cmdf(cmddir, bf, age, logz, vvc, av, dmod):

    """
        Returns the .out.cmd file of a given set of parameters from a calcsfh run. These files 
    contain the MATCH constructed Hess diagrams.
    """

    vvc = str(vvc)
    #age = str(age)

    # for the path to the .cmd file:
    cmdpath = os.path.join(cmddir,
                                   'bf{:s}_t{:.2f}_logz{:s}_vvc{:s}_av{:s}_dmod{:s}*.out.cmd'.format(bf, age,
                                                                                                     logz, vvc,
                                                                                                     av, dmod)
                                                                                                     )

    #print("Looking for {:s}...".format(cmdpath))
    cmdfn = glob.glob(cmdpath)[0]

    return cmdfn