import numpy as np
import matplotlib.pyplot as plt
import pickle
import math
import os
from scipy.optimize import leastsq
import sys
from fou import ft, invft  # where the business happens

# start of 'main'
param = sys.argv
lambda_bar = float(param[1])

gamma = param[2]
if "." in gamma:
    gamma = float(gamma)
else:
    gamma = int(gamma)
path = param[3]

where_to_save = path

# depickling
with open(
    "{}values_lambda={}_gamma={}_half.pickle".format(where_to_save, lambda_bar, gamma),
    "rb",
) as f:
    things = pickle.load(f, encoding="latin1")
info = things[0]
phi = things[1]
z = things[2]
lambda_bar, gamma, d, ds, n_z, how_often_ds, r_info = info
r0 = r_info[0]
dz = abs(z[1] - z[0])

M = np.sqrt(1 / 18 * (9 - 8 * lambda_bar + 3 * np.sqrt(9 - 8 * lambda_bar)))
phi_b = (1 + np.sqrt(1 - 8 / 9 * lambda_bar)) / 2
s = np.linspace(0, (len(phi[:, 0]) - 1) * ds * how_often_ds, len(phi[:, 0]))

# where to start from
s_col = np.sqrt((d / 2) ** 2 - r0**2)
phi = phi[s >= s_col]
s = s[s >= s_col]

# where to end
s_end = np.sqrt(d**2 - r0**2)
phi = phi[s <= s_end]
s = s[s <= s_end]

# down-sampling
every_s = 100
every_z = 100
phi = phi[::every_s, ::every_z]
s = s[::every_s]
z = z[::every_z]
ds = ds * every_s * how_often_ds
dz = dz * every_z

# a simple check
if abs(ds - (s[1] - s[0])) > 1.0e-12:
    print("ds error: " + str(ds) + " != " + str(s[1] - s[0]))
    sys.exit()
if abs(dz - (z[1] - z[0])) > 1.0e-12:
    print("dz error: " + str(dz) + " != " + str(z[1] - z[0]))
    sys.exit()

N_s = len(phi[:, 0])
N_z = len(phi[0])
L_s = (N_s - 1) * ds

# the smallest values of k and w
dk = np.pi / dz / (N_z - 1.0)
dw = np.pi / ds / N_s

fourierlist = np.zeros((N_s, N_z))

# the fourier transformation:
# a combination of sinc transform in s
# and type-I cos transform in z
ft(fourierlist, phi, N_s, N_z)

# saving the result
to_dump = (fourierlist, dk, dw, N_z, N_s)
with open(
    "{}fourier_lambda_{}_gamma_{}_everys_{}_everyz_{}.pickle".format(
        where_to_save, lambda_bar, gamma, every_s, every_z
    ),
    "wb",
) as f:
    pickle.dump(to_dump, f)

# checking the inverse
check_inverse = True
if check_inverse:
    # inverse fourier transform
    phi2list = np.zeros((N_s, N_z))
    invft(phi2list, fourierlist, N_s, N_z)

    error = phi - phi2list

    error_norm = 0.0
    phi_norm = 0.0
    for j_s in range(N_s):
        for j_z in range(N_z):
            error_norm += abs(error[j_s, j_z])
            phi_norm += abs(phi[j_s, j_z])

    print("norm(FT[F[phi]]-phi)/norm(phi) = {}".format(error_norm / phi_norm))
