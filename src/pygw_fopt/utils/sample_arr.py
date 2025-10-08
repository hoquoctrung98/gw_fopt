from .sample import sample
import numpy as np

def sample_arr(xmin, xmax, nsample, ngrid, niter, type="linear", base=10):
    arr = np.concatenate([
        sample(xmin, xmax, nsample, ngrid, niter=i, type=type, base=base) for i in range(niter + 1)
    ])
    return arr
