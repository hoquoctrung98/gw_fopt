import numpy as np

def linsample(xmin, xmax, nsample, ngrid, niter):
    xrange = np.abs(xmax - xmin)
    if niter == 0:
        arr = np.linspace(xmin, xmax, nsample + 1)
    else:
        arr = np.array([
            xmin + i1 * xrange / (nsample * (ngrid ** niter)) +
            i2 * xrange / (nsample * (ngrid ** (niter - 1)))
            for i2 in range(0, nsample * (ngrid ** (niter - 1)))
            for i1 in range(1, ngrid)
        ])
    return arr
