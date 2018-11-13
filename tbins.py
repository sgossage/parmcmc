#!/usr/bin/env python

import numpy as np
import sys

def frange(start, stop, step, N=None):

    """
        Meant to generate an (numpy) array of 
    floats that avoids floating point errors.
    E.g., give start = 0, stop = 10.0,, step=0.1 
    this should give an array of [0.0, 0.1, 0.2, 
    0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 10.0]. 
    """

    curval = start
    if N is None:
        numele = int(round((stop - start)/step))
    else:
        numele = N

    arr = np.zeros(numele+1)

    i = 0
    while curval <= stop:

        arr[i] = curval
        curval += step
        i += 1
    
    return arr


def rwparamf(paramfn, t0, t1, tbin, savefn):

    t0 = round(float(t0), 6)
    t1 = round(float(t1), 6)
    tbin = round(float(tbin), 2)

    print(t0)
    print(t1)
    print(tbin)

    #N = int(round((t1 - t0) / tbin))
    #print(N)
    #if N == 1:
    #    N = 2

    ages = frange(t0, t1, tbin) #np.linspace(t0, t1, N)
    print(ages)
    # the actual number of bins is one less 
    # than the number of ages, since e.g., 
    # there's no bin for the last age.
    N = len(ages) - 1

    with open(paramfn, 'r') as paramfi:
        lines = paramfi.readlines()

    prevN = int(lines[8])
    lines[8] = '{:d}\n'.format(N)
    leftovers = lines[9+prevN::]
    print(leftovers)

    for i in range(max(prevN+1, N+1)):
        if i == N:
            # break if on last age. Don't write an age bin for the last age (no final age + x value).
            print('!')
            break
        try:
            # re-writing a line
            lines[9+i] = "   {:.6f} {:.6f}\n".format(round(ages[i], 2), round(ages[i+1], 2))
        except IndexError:
            # or adding a new line
            lines.append("   {:.6f} {:.6f}\n".format(round(ages[i], 2), round(ages[i+1], 2)))
            print('!!')
            #break

    # cut lines beyond what's desired
    lines = lines[:9+i:]
    # tack on end of file contents
    lines = lines + leftovers

    print(lines)

    with open(savefn, 'w') as paramfo:
        paramfo.writelines(lines)

    return


if __name__ == '__main__':

    paramfn, t0, t1, tbin, savefn = sys.argv[1::]

    # re-write the calcsfh parameter file:
    rwparamf(paramfn, t0, t1, tbin, savefn)
        

    
