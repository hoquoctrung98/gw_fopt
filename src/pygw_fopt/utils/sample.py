from .linsample import linsample
from .logsample import logsample

def sample(xmin, xmax, nsample, ngrid, niter, type="linear", base=10):
    if type == "linear":
        return linsample(xmin, xmax, nsample, ngrid, niter)
    elif type == "log":
        return logsample(xmin, xmax, nsample, ngrid, niter, base)
