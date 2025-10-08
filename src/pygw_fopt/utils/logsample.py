import numpy as np

def logsample(xmin, xmax, nsample, ngrid, niter, base=10):
    log_xmin = np.log10(xmin) / np.log10(base)
    log_xmax = np.log10(xmax) / np.log10(base)
    log_xrange = log_xmax - log_xmin
    if niter == 0:
        arr = np.logspace(log_xmin, log_xmax, nsample + 1, base=base)
    else:
        arr = base**np.array([
            log_xmin + i1 * log_xrange / (nsample * (ngrid ** niter)) +
            i2 * log_xrange / (nsample * (ngrid ** (niter - 1)))
            for i2 in range(0, nsample * (ngrid ** (niter - 1)))
            for i1 in range(1, ngrid)
        ])
    return arr
