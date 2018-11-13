from tbins import rwparamf
from tbins import frange
import sys
import numpy as np

def mk_agegrid(paramfn, t0, tf, tbin):

    """
       Give this script a start and end log age and it 
    will create a set of calcsfh parameter files with 
    delta age (i.e. a single age bin) at each age in the 
    range of t0 to tf in tbin steps. These are created 
    from a template calcsfh param file whose name is 
    paramfn (which will hold all the other calcsfh 
    parameter specifications).

    """

    #paramfn, t0, tf, tbin = sys.argv[1::]

    t0 = round(float(t0), 2)
    tf = round(float(tf), 2)
    tbin = round(float(tbin), 2)

    #N = (tf - t0) / tbin

    trange = frange(t0, tf, tbin)#np.linspace(t0, tf, N)

    paramfn_base = paramfn.split('.param')[0]

    for age in trange:
        if age == trange[-1]:
            break

        age = round(age, 2)
        savefn = "{:s}_t{:.2f}.param".format(paramfn_base, age)
        print(age)
        print(age+tbin)
        rwparamf(paramfn, age, age+tbin, tbin, savefn)

    return
