import numpy as np
import math


def solid_angle(D):
    return 2.0 * (np.pi ** (D / 2)) / math.gamma(D)
