import faulthandler

faulthandler.enable()
import numpy as np
import matplotlib.pyplot as plt
import math
import os
import sys
from scipy.special import jv
import cmath
from scipy.optimize import leastsq
from scipy.interpolate import interp1d
import scipy.integrate
import pickle
import gw_integrator

# defining functions


# writing constants to file
def write_constants_to_file(name, value):
    if not different_spacing:
        file = open(
            "{}constants_lambda={}_gamma={}_half.txt".format(
                path_to_data, lambda_bar, gamma
            ),
            "a",
        )
    elif different_spacing:
        file = open(
            "{}constants_lambda={}_gamma={}_half_dztimes{}_ds={}.txt".format(
                path_to_data, lambda_bar, gamma, dz_times, ds
            ),
            "a",
        )

    file.write(name + "\t" + str(value) + "\n")
    file.close()
    print(name, flush=True)


# writing data to file
def write_to_file(w, dE_dlogw):
    if not different_spacing:
        file = open(
            "{}gw_results_whole_lambda={}_gamma={}_w{}.txt".format(
                path_to_data, lambda_bar, gamma, current_w_num
            ),
            "a",
        )
    elif different_spacing:
        file = open(
            "{}gw_results_whole_lambda={}_gamma={}_dztimes{}_ds={}_w{}.txt".format(
                path_to_data, lambda_bar, gamma, dz_times, ds, current_w_num
            ),
            "a",
        )
    file.write(str(w) + "\t" + str(dE_dlogw) + "\n")
    file.close()
    print((w), flush=True)
    print((dE_dlogw), flush=True)


# printing important info of the run to the terminal or slurm file and the constants file
def print_info():
    print("svals =", len(phi))
    print("zvals =", len(phi[0]))
    print("z0 =", z[0])
    print("gamma = " + str(gamma))
    print("lambdabar = " + str(lambda_bar))
    print("ds = " + str(ds))
    print("dz = " + str(dz))
    print("d = " + str(d))
    r0, r_in, r_out = r_info
    print("r0 = " + str(r0))
    print("r_in = " + str(r_in))
    print("r_out = " + str(r_out))
    print("simulation smax = " + str(smax))
    print("cutoff_ratio = " + str(cutoff_ratio))
    print("n_s = " + str(len(slist)))
    print("n_z = " + str(len(z)), flush=True)

    # creating the constants file
    if current_w_num == 0:
        write_constants_to_file("lambdabar", lambda_bar)
        write_constants_to_file("gamma", gamma)
        write_constants_to_file("ds", ds)
        write_constants_to_file("dz", dz)
        write_constants_to_file("d", d)
        write_constants_to_file("r0", r0)
        write_constants_to_file("r_in", r_in)
        write_constants_to_file("r_out", r_out)
        write_constants_to_file("n_s", n_s)
        write_constants_to_file("n_z", n_z)
        write_constants_to_file("how_often_ds", how_often_ds)
        write_constants_to_file("cutoff_ratio", cutoff_ratio)


# creating the phi for the region r^2>t^2
def phi_region2(phi0, d, slist, z, type="half"):

    n_s = len(slist)
    n_z = len(z)
    ds = slist[1] - slist[0]
    dz = z[1] - z[0]

    # if there's only one bubble
    if type == "one":
        # interpolating the values from the original bubble profile
        phi0_interp = interp1d(z, phi0, kind="linear", fill_value="extrapolate")
        phi2 = np.fromfunction(
            lambda i_s, i_z: phi0_interp(np.sqrt((i_s * ds) ** 2 + (i_z * dz) ** 2)),
            (n_s, n_z),
        )

    # if there's two bubbles
    else:
        # finding the original bubble profile from the first slice
        phimid = np.argmax(phi0)
        z = z[phimid::]
        print(z[0], d / 2)
        z = z - z[0]
        phi0 = phi0[phimid::]

        # interpolating the values from the original bubble profile
        phi0_interp = interp1d(z, phi0, kind="linear", fill_value="extrapolate")
        phi2 = np.fromfunction(
            lambda i_s, i_z: (
                phi0_interp(np.sqrt((i_s * ds) ** 2 + (i_z * dz - d / 2) ** 2))
                + phi0_interp(np.sqrt((i_s * ds) ** 2 + (i_z * dz + d / 2) ** 2))
            ),
            (n_s, n_z),
        )

    return phi2


# number of steps for k
n_k = 51
# default cutoff ratio
cutoff_ratio = 0.9

# reading run parameters

# parameters from command line arguments
param = sys.argv

# is k integrand plotted?
plot_k = False
if "plot_k" in param:
    plot_k = True
    param.remove("plot_k")

# are the sub integrals printed?
print_sub = False
if "print_sub" in param:
    print_sub = True
    param.remove("print_sub")

# setting parameters
lambda_bar = float(param[1])

gamma = param[2]
if "." in gamma:
    gamma = float(gamma)
else:
    gamma = int(gamma)

current_w_num = int(param[3])
n_w = int(param[4])
path_to_data = param[5]

try:
    name = param[6]
    if name != "one" or name != "half":
        name = "half"
except:
    name = "half"

# in case of non automatic spacing choice for hyper_bubbles.py
try:
    dz_times = param[7]
    ds = param[8]
    different_spacing = True
except:
    different_spacing = False


# taking data from pickle file
if not different_spacing:
    with open(
        "{}values_lambda={}_gamma={}_{}.pickle".format(
            path_to_data, lambda_bar, gamma, name
        ),
        "rb",
    ) as f:
        things = pickle.load(f, encoding="latin1")
elif different_spacing:
    with open(
        "{}values_lambda={}_gamma={}_{}_dztimes{}_ds={}.pickle".format(
            path_to_data, lambda_bar, gamma, name, dz_times, ds
        ),
        "rb",
    ) as f:
        things = pickle.load(f, encoding="latin1")

info = things[0]
phi = things[1]
z = things[2]
del things
lambda_bar, gamma, d, ds_orig, n_z, how_often_ds, r_info = info


ds = ds * how_often_ds
dz = abs(z[1] - z[0])
n_z = len(z)

M = np.sqrt(1 / 18 * (9 - 8 * lambda_bar + 3 * np.sqrt(9 - 8 * lambda_bar)))

# calculating the range for w and the w value for this run
wmin = np.pi / ((len(phi[0]) - 1) * dz) / 2  # pi/(size of the box)print(10*M, np.pi/dz)
wmax = min(10 * M, np.pi / dz)
wlist = np.geomspace(wmin, wmax, n_w)
w = wlist[current_w_num]

# finding smax and creating slist
smax = (len(phi) - 1) * ds
slist = np.linspace(0, smax, len(phi))
n_s = len(phi)

# creating the second region phi
phi2 = phi_region2(phi[0], d, slist, z, name)

print_info()


def plot_k_integrand(klist, intk):
    plt.ylabel("integrand (Eq. 32)")
    plt.xlabel("k")
    plt.plot(klist, intk)
    plt.savefig(
        "{}k_integrand_lambdabar={}_gamma={}_w={}.txt".format(
            path_to_data, lambda_bar, gamma, w
        )
    )


# running the code in gw_integrator.py
w, int_k, klist, intk = gw_integrator.gw_integral(
    current_w_num, n_k, w, z, phi, phi2, ds, path_to_data, cutoff_ratio, print_sub
)

write_to_file(w, int_k)

if plot_k:
    plot_k_integrand(klist, intk)
