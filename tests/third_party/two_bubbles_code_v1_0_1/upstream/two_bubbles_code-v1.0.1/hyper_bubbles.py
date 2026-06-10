from cosmoTransitions.tunneling1D import SingleFieldInstanton
import matplotlib.pyplot as plt
import math
import numpy as np
from scipy.interpolate import interp1d
import os
import matplotlib
import pickle
import time
import sys
import itertools
import csv


# definitions of potential and parameters
def V(phi, lambdabar):
    return phi**2 * (-phi / 3 + phi**2 / 4 + lambdabar / 9)


def dV(phi, lambdabar):
    return phi**3 - phi**2 + 2 / 9 * lambdabar * phi


phi_false = 0
alpha = 3
ds = 0.01  # maximum ds, changed automatically to smaller if needed
dz_max = 0.05  # maximum dz, changed automatically to smaller if needed
how_often_ds = 5  # how often in s is saved

dz_mult = 1
set_ds = False

############################################################
# simulation parameters
param = sys.argv
print(param)

lambdabar = float(param[1])
gamma = param[2]
if "." in gamma:
    gamma = float(gamma)
else:
    gamma = int(gamma)

where_to_save_data = param[3]


############################################################

############################################################
# additional parameters
try:
    p = param[4]
    try:
        dz_mult = float(p)
        test = False
        try:
            ds_hard = float(param[5])
            set_ds = True
        except:
            set_ds = False
    except:
        dz_mult = 0.5

    if p == "video":
        video = True
        test = False
    elif not test:
        video = False
    else:
        test = True
        video = False
        which_test = p
except:
    test = False
    video = False
############################################################

# field value in broken phase
phi_b = (1 + np.sqrt(1 - 8 / 9 * lambdabar)) / 2


# nucleates a bubble wth cosmotransitions
def nucleation(absmin, falsemin, alpha):
    bubble = SingleFieldInstanton(
        absmin,
        falsemin,
        lambda phi: V(phi, lambdabar),
        lambda phi: dV(phi, lambdabar),
        alpha=alpha,
    )
    return bubble


# returns the field of two bubbles
def two_bubbles(absmin, falsemin, alpha, gamma, type="half"):
    # header
    print("\ninitial conditions:")

    # amount of points to get from cosmotransitions
    npoints = 1000

    # first bubble
    bubble1 = nucleation(phi_b, phi_false, alpha)
    profile1 = bubble1.findProfile(npoints=npoints)
    phi1 = profile1.Phi
    r1 = profile1.R

    # calculating the original, outer and inner radiuses respectively
    phi00 = phi1[0]
    index0 = (np.abs(phi1 - phi00 / 2)).argmin()
    r0 = r1[index0]

    phi_r_out = phi00 / 2 * (1 - np.tanh(1 / 2))
    index_out = (np.abs(phi1 - phi_r_out)).argmin()
    r_out = r1[index_out]

    phi_r_in = phi00 / 2 * (1 - np.tanh(-1 / 2))
    index_in = (np.abs(phi1 - phi_r_in)).argmin()
    r_in = r1[index_in]

    # calculating the distance of the two bubbles from gamma and r0
    d = 2 * gamma * r0

    # adding extra values to the lists for extrapolation
    phi1 = np.append(phi1, [0, 0])
    r1 = np.append(r1, r1[-1] + r1[1] - r1[0])
    r1 = np.append(r1, d)

    # second bubble, flipped and moved from origin
    bubble2 = nucleation(phi_b, phi_false, alpha)
    profile2 = bubble2.findProfile(npoints=npoints)
    phi2 = profile2.Phi
    phi2 = np.append(phi2, np.array([0, 0]))
    phi2 = phi2[::-1]

    r2 = profile2.R
    r2 = np.append(r2, r2[-1] + r2[1] - r2[0])
    r2 = np.append(r2, d)
    r2 = -r2[::-1] + d

    # wall width at collision, used to calculate dz
    l_wall_hit = np.sqrt(r_out**2 + (d / 2) ** 2 - r0**2) - np.sqrt(
        r_in**2 + (d / 2) ** 2 - r0**2
    )

    # getting at least 10 values within the wall at collision
    # however, dz will be no larger than dz_max
    dz_approx = l_wall_hit / 10
    if dz_approx > dz_max:
        dz_approx = dz_max
    dz_approx *= dz_mult
    # printing gamma calculated with lorentz contraction
    gamma_c = (r_out - r_in) / l_wall_hit
    print("gamma_c   = " + str(gamma_c))

    # amount of points in z axis
    n_z = int(round(d / dz_approx))

    # let's take care that there's a specific middle point
    if n_z % 2 == 0:
        n_z += 1

    # creating the lattice in z
    z0 = np.linspace(r1[0], r1[-1], n_z)

    # interpolating the fields of each bubble and
    phi1_interpl = interp1d(r1, phi1, kind="linear")
    phi2_interpl = interp1d(r2, phi2, kind="linear")

    phi1 = phi1_interpl(z0)
    phi2 = phi2_interpl(z0)

    # combining bubble fields
    for i in range(len(phi1)):
        phi1[i] = np.sqrt(phi1[i] ** 2 + phi2[i] ** 2)
    print("max(phi)  = " + str(np.amax(phi1)))

    # middle range values
    z0_mid_range = z0
    phi_mid_range = phi1

    # the left side of the bubble is the same as the field of the flipped bubble
    # for the z axis we have to flip it and make it negative
    z0_left = z0 - z0[-1]
    phi_left = phi2

    # now we have the field between the bubbles
    # however, we want to take points only up the half point (1,...,n_z/2+1)
    # let's also remove the bubble center (first) point from phi and z
    # so it won't be there twice
    phi1 = phi1[1 : int((n_z - 1) / 2) + 1]
    z0_half = z0[1 : int((n_z - 1) / 2) + 1]
    z_width = 2 * z0_half[-1]
    # print('d         = ' + str(z_width))

    # the left side of the bubble is the same as the field of the flipped bubble
    # for z axis we have to remove the fist point, flip it and make it negative

    # let's combine these two
    z0 = np.append(z0_left, z0_half)
    phi = np.append(phi_left, phi1)

    phi = phi[::-1]
    z0 = -z0[::-1] + d / 2
    dz = z0[1] - z0[0]

    # in order to keep the bubble from hitting the edge, lets add some empy space
    extralength = d * 0.3
    extranum = int(round(extralength / dz))  # amount of lattice points added
    phiextra = np.zeros(extranum + 1)  # added field values are 0
    extraz = np.linspace(0, dz * extranum, extranum + 1) + z0[-1] + dz
    z0 = np.append(z0, extraz)
    phi = np.append(phi, phiextra)

    print("zmax      = " + str(np.amax(z0)))
    print("zmax/d    = " + str(np.amax(z0) / d))

    # returning the desired field profile

    if type == "one":  # returns one bubble profile
        z0_left = -z0_left[::-1]
        phi_left = phi_left[::-1]

        # addind room to expand
        phiextra = np.zeros(extranum + 1)
        extraz = np.linspace(0, dz * extranum, extranum + 1) + z0_left[-1] + dz
        z0_left = np.append(z0_left, extraz)
        phi_left = np.append(phi_left, phiextra)

        print("one bubble")
        return z0_left, phi_left, len(z0_left), d, (r0, r_in, r_out), phi00

    if type == "whole":  # returns the whole field
        otherhalfz = -z0[1::]
        otherhalfphi = phi[1::]
        wholez = np.append(otherhalfz[::-1], z0)
        wholephi = np.append(otherhalfphi[::-1], phi)

        print("whole field")
        return wholez, wholephi, len(wholez), d, (r0, r_in, r_out)

    if type == "mid":  # returns middle section of the field
        print("mid part of the field")
        nz_mid_range = len(z0_mid_range)
        return z0_mid_range, phi_mid_range, nz_mid_range, d, (r0, r_in, r_out)

    # returns half of the whole field (mirror symmetry in z)
    print("half of whole field")
    n_z_half = len(z0)
    return z0, phi, n_z_half, d, (r0, r_in, r_out)


# calculating the next derivative of phi, pi=dphi/ds
def evolvepi(phi, pi, s, ds, dz):
    pilist = np.empty(len(phi))
    dV_list = dV(phi, lambdabar)
    for i in range(len(phi)):
        # taking care of boundary conditions
        if i != 0:  # left boundary
            phiprev = phi[i - 1]
        else:
            phiprev = phi[1]
        if i != len(phi) - 1:  # right boundary
            phinext = phi[i + 1]
        else:
            phinext = phi[-2]

        pilist[i] = pi[i] * (1 - 2 * ds / (s + ds)) + s * ds / (s + ds) * (
            -dV_list[i] + (phiprev - 2 * phi[i] + phinext) / (dz) ** 2
        )

    return pilist


# finding pi=dphi/ds for the first half step in leapfrog
def evolvepi_first_half_step(phi, ds, dz, baby_steps):
    pi = np.zeros(len(phi))  # initially pi=dphi/ds=0
    baby_ds = (0.5 * ds) / (baby_steps - 1)
    for i in range(1, baby_steps):
        pi = evolvepi(phi, pi, (i - 1) * baby_ds, baby_ds, dz)
        phi = phi + baby_ds * pi

    return pi


# evolving field n times with leapfrog
def evolve(phi, ds, dz, n):
    phicomplete = phi
    baby_steps = 100
    pi = evolvepi_first_half_step(phi, ds, dz, baby_steps)
    n_print = n // 20

    print("\nevolving field:", flush=True)
    print("%-8s %-8s %-12s" % ("%", "i_s", "s"), flush=True)
    for i in range(1, n + 1):
        if i % n_print == 0:
            print("%-8.0f %-8d %-12.4f" % (100 * i / n, i, i * ds), flush=True)
        if i > 1:
            pi = evolvepi(phi, pi, (i - 1) * ds, ds, dz)
        phi = phi + ds * pi
        phicomplete = np.vstack((phicomplete, phi))
    print("", flush=True)

    return phicomplete


# plotting phi as series of images
def plot_phi(z, phiend, how_often, n, ds, dz):
    until = round(n / how_often)
    phiend_scaled = phiend / phi_b
    min = int(np.amin(phiend_scaled))
    if min > 0:
        min = 0
    else:
        min = math.floor(2 * np.amin(phiend_scaled)) / 2
    max = math.ceil(2 * np.amax(phiend_scaled)) / 2

    # plotting n figures that can be made into a video
    for i in range(until):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(z, phiend_scaled[i * how_often])

        # labels
        ax.set_xlabel(r"z")
        ylabel = ax.set_ylabel(r"$\frac{\varphi}{\varphi_b}$")
        ylabel.set_rotation(0)
        txt_l = r"$ \bar{ \lambda }$"
        ax.set_title(
            r"{}={}, $\gamma$={}, ds={}, dz={}".format(
                txt_l, lambdabar, gamma, round(ds, 4), round(dz, 4)
            )
        )
        ax.axis([0, np.amax(z), 0, max])

        # makes a new directory if it doesn't exixt
        if (
            os.path.exists(
                "{}plots/phi_plot_lambda={}_gamma={}".format(
                    where_to_save_data, lambdabar, gamma
                )
            )
            == False
        ):
            os.makedirs(
                "{}plots/phi_plot_lambda={}_gamma={}".format(
                    where_to_save_data, lambdabar, gamma
                )
            )

        fig.savefig(
            "{}plots/phi_plot_lambda={}_gamma={}/phi_plot_{:04n}.png".format(
                where_to_save_data, lambdabar, gamma, i
            ),
            dpi=200,
        )
        plt.close("all")


# plotting a 2D plot
def plot_color(z, phiend, ds, n):
    phiend_scaled = phiend / phi_b
    ns = len(phiend_scaled[:, 0])
    s = np.arange(0, ds * ns, ds)

    plt.imshow(
        phiend_scaled[::-1],
        cmap="RdBu_r",
        extent=[np.amin(z), np.amax(z), 0, np.amax(s)],
    )

    plt.xlabel("z")
    yl = plt.ylabel("s")
    yl.set_rotation(0)
    plt.title(r"$ \bar{ \lambda }$ = " + str(lambdabar))
    cbar = plt.colorbar()
    cbar.set_label(r"$\frac{\varphi}{\varphi_b}$", rotation=0)
    plt.clim(1 - np.amax(phiend_scaled), np.amax(phiend_scaled))

    if dz_mult != 1 or set_ds:
        plt.savefig(
            "{}gamma={}_lambdabar={}.png".format(where_to_save_data, gamma, lambdabar)
        )
    else:
        plt.savefig(
            "{}gamma={}_lambdabar={}_dztimes{}_ds={}.png".format(
                where_to_save_data, gamma, lambdabar, dz_mult, ds
            )
        )

    plt.show()


# calculating energy density every tenth s-point
def calculate_energy_density(phiall, ds, dz):
    n_z = len(phiall[0])
    n_s = len(phiall[:, 0])
    num = len(range(1, n_s - 1, 10))
    energy_density = np.zeros((num, n_z - 1))

    for index, i in enumerate(range(1, n_s - 1, 10)):
        for j in range(n_z - 1):
            energy_density[index][j] = (
                1 / 2 * ((phiall[i + 1, j] - phiall[i - 1, j]) / (2 * ds)) ** 2
                + 1 / 2 * ((phiall[i, j + 1] - phiall[i, j]) / (dz)) ** 2
                + V(phiall[i, j], lambdabar)
            )
    return energy_density


# calculating the complete energy at time s
def energy(phiall, ds, dz):
    elist = []
    n_z = len(phiall[0])
    n_s = len(phiall[:, 0])

    for i in range(1, n_s - 1):
        e_i = 0
        for j in range(n_z):
            if j == 0:
                kin_s = (
                    0.5
                    * dz
                    * 1
                    / 2
                    * ((phiall[i + 1, j] - phiall[i - 1, j]) / (2 * ds)) ** 2
                )
                kin_z = dz * 1 / 2 * ((phiall[i, j + 1] - phiall[i, j]) / (dz)) ** 2
                pot = 0.5 * dz * V(phiall[i, j], lambdabar)
            elif j == (n_z - 1):
                kin_s = (
                    0.5
                    * dz
                    * 1
                    / 2
                    * ((phiall[i + 1, j] - phiall[i - 1, j]) / (2 * ds)) ** 2
                )
                kin_z = 0
                pot = 0.5 * dz * V(phiall[i, j], lambdabar)
            else:
                kin_s = (
                    dz * 1 / 2 * ((phiall[i + 1, j] - phiall[i - 1, j]) / (2 * ds)) ** 2
                )
                kin_z = dz * 1 / 2 * ((phiall[i, j + 1] - phiall[i, j]) / (dz)) ** 2
                pot = dz * V(phiall[i, j], lambdabar)
            e_i += kin_s + kin_z + pot
        elist = np.append(elist, e_i)

    return elist


# plotting energy in s
def plot_energy(e, ds, dz):
    s = ds * np.array(range(len(e)))
    plt.plot(s, e)
    plt.xlabel("s")
    ylabel = plt.ylabel("E")
    ylabel.set_rotation(0)
    plt.title("ds=" + str(round(ds, 4)) + ", dz=" + str(round(dz, 5)))
    plt.tight_layout()
    plt.savefig(
        "{}E(s)_lambdabar={}_gamma={}.png".format(where_to_save_data, lambdabar, gamma)
    )
    plt.show()


# Kosowsky, eq. 44, LHS
def energy_derivative1(e, ds):
    dE = np.array([])
    slist = np.array([])
    for i in range(len(e) - 1):
        de = (e[i + 1] - e[i]) / ds
        dE = np.append(dE, de)
        slist = np.append(slist, ds * (i + 1.5))
    return dE, slist


# Kosowsky, eq. 44, RHS
def energy_derivative2(phi, ds, dz):
    dE = np.array([])
    slist = np.array([])
    n_z = len(phi[0])
    n_s = len(phi[:, 0])
    for i in range(1, n_s - 2):
        s = ds * (i + 0.5)
        de = 0
        for j in range(n_z):
            if j == 0 or j == (n_z - 1):
                de += -0.5 * 2 * dz * 1 / s * ((phi[i + 1, j] - phi[i, j]) / (ds)) ** 2
            else:
                de += -2 * dz * 1 / s * ((phi[i + 1, j] - phi[i, j]) / (ds)) ** 2
        dE = np.append(dE, de)
        slist = np.append(slist, s)
    return dE, slist


# plotting the comparison between the RHS and LHS of Kosowsky eq. 44
def plot_energy_derv_comp(dE1, slist1, dE2, slist2, lambdabar, gamma, ds):
    plt.plot(slist1, dE1, color="r", label="LHS")
    plt.plot(slist2, dE2, color="b", label="RHS")

    plt.xlabel(r"$s$")
    yla = plt.ylabel(r"$\frac{\mathrm{d}E}{\mathrm{d}s}$", y=0.49, fontsize=18)
    yla.set_rotation(0)
    plt.title(
        r"$\frac{1}{2 \pi} \frac{\mathrm{d}E}{\mathrm{d}s} =$"
        + r"$-\frac{2}{s} \int \ dz (\frac{\partial \phi}{\partial s})^2$",
        y=1.03,
        fontsize=18,
    )
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        "{}energy_derivative_comparison(s)_lambda={}_gamma={}_ds={}.png".format(
            where_to_save_data, lambdabar, gamma, ds
        )
    )
    plt.show()


# testing energy derivative of specific ds and plotting it in s
def energy_derivative_test(gamma, ds, ds_set):

    how_often = 25
    print(gamma)
    # finding phi(z) that evolves in s:
    z, phi, n_z, d, rinfo = two_bubbles(phi_b, phi_false, alpha, gamma)
    dz = abs(z[1] - z[0])
    # calculating the number of times the field is evolved
    smax = d * 1.2
    n = int(round((smax) / ds))
    slist = np.linspace(0, smax, n + 1)
    ds = slist[1] - slist[0]

    # if dz is smaller than ds, let's make ds smaller
    if dz <= ds and (not ds_set):
        ds = dz / 2

    # calculating the number of times the field is evolved
    smax = d * 1.2
    n = int(round((smax) / ds))
    slist = np.linspace(0, smax, n + 1)
    ds = slist[1] - slist[0]

    # printing important info
    print("gamma     = " + str(gamma))
    print("lambdabar = " + str(lambdabar))
    print("r_0       = " + str(r_info[0]))
    print("d         = " + str(d))
    print("ds        = " + str(ds))
    print("dz        = " + str(dz))
    print("n_s       = " + str(n))
    print("n_z       = " + str(n_z))
    print("dz        = " + str(dz), flush=True)

    # simulation
    phiend = evolve(phi, ds, dz, n)

    plot_color(z, phiend, ds, n)

    # calculating E(s)
    e = energy(phiend, ds, dz)
    plot_energy(e, ds, dz)

    dE1, slist1 = energy_derivative1(e, ds)
    dE2, slist2 = energy_derivative2(phiend, ds, dz)

    # comparing derivatives of energy

    fig, [ax1, ax2] = plt.subplots(
        nrows=1, ncols=2, figsize=(13, 5), constrained_layout=True
    )

    ax1.plot(slist1, dE1, color="r", label="LHS")
    ax1.plot(slist2, dE2, color="b", label="RHS")
    print(len(dE1), len(dE2))

    ax1.set_xlabel(r"$s$")
    yla = ax1.set_ylabel(r"$\frac{\mathrm{d}E}{\mathrm{d}s}$", y=0.49, fontsize=18)
    yla.set_rotation(0)
    ax1.set_title(
        r"$\frac{1}{2 \pi} \frac{\mathrm{d}E}{\mathrm{d}s} =$"
        + r"$-\frac{2}{s} \int dz (\frac{\partial \phi}{\partial s})^2$",
        y=1.03,
        fontsize=18,
    )
    ax1.legend()

    slist = slist1
    print(len(dE2))

    norm = 1 / 2 * (np.sum(dE1) / len(dE1) + np.sum(dE2) / len(dE2))
    dde = (dE1 - dE2) / norm
    print(norm)
    ax2.plot(slist, dde)
    yla = ax2.set_ylabel(r"$(\partial_s E_1-\partial_s E_2)$/norm")
    ax2.set_title(r"Normalised difference of energy derivatives", y=1.03)
    ax2.set_xlabel(r"$s$")
    plt.savefig(
        "{}energy_density_difference_lambda={}_gamma={}.png".format(
            where_to_save_data, lambdabar, gamma
        )
    )

    to_save = itertools.zip_longest(dde, slist, dE1, dE2, e)
    print(to_save)
    with open(
        "{}energy_derivative_difference_lambda={}_gamma={}.txt".format(
            where_to_save_data, lambdabar, gamma
        ),
        "w",
    ) as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["dde", "s", "dE_LHS", "dE_RHS", "energy"])
        writer.writerows(to_save)

    plt.show()


# testing energy derivative for different ds (in range(lower, upper, gap))
def energy_derivative_convergence_test(lower, upper, gap, gamma):

    e_change_list = np.array([])
    dslist = np.array([])

    # creating the bubbles
    z, phi, n_z, d, rinfo = two_bubbles(phi_b, phi_false, alpha, gamma)
    dz = abs(z[1] - z[0])

    # calculating differences is energy derivatives for different ds
    for ds in np.arange(lower, upper, gap):
        # calculating the number of times the field is evolved
        smax = d * 1.2
        n = int(round((smax) / ds))
        slist = np.linspace(0, smax, n + 1)
        ds = slist[1] - slist[0]
        print(ds)

        phiend = evolve(phi, ds, dz, n)
        e = energy(phiend, ds, dz)
        dslist = np.append(dslist, ds)

        # calculating the LHS and RHS of the energy derivative
        dE1, slist1 = energy_derivative1(e, ds)
        dE2, slist2 = energy_derivative2(phiend, ds, dz)
        slist = slist1
        # making the lists the same size
        plot_energy_derv_comp(dE1, slist1, dE2, slist2, lambdabar, gamma, ds)

        e = np.delete(e, 0)

        norm = 1 / 2 * (np.sum(dE1) / len(dE1) + np.sum(dE2) / len(dE2))
        # difference of energy derivates in dimensionless units
        dde = (dE1 - dE2) / norm
        plt.plot(slist, dde)
        yla = plt.ylabel(
            r"$ \frac{\mathrm{d}E1-\mathrm{d}E2}{\mathrm{norm}}$", fontsize=18
        )
        yla.set_rotation(0)
        plt.xlabel(r"$s$")
        plt.show()

        # largest difference of enery derivates
        Dde = np.abs(np.amax(dde))
        print(Dde)
        e_change_list = np.append(e_change_list, Dde)

    # writing values to a file
    to_dump = (e_change_list, dslist)
    with open(
        "{}dE(ds)_values_lambda={}_gamma={}_nz={}.pickle".format(
            where_to_save_data, lambdabar, gamma, n_z
        ),
        "wb",
    ) as f:
        pickle.dump(to_dump, f)

    log_e_change = np.log(e_change_list)
    log_ds = np.log(dslist)
    slope_fit, const_fit = np.polyfit(log_ds, log_e_change, 1)

    x = np.linspace(np.amin(dslist), np.amax(dslist), 1000)
    x_log = np.log(x)

    fig, [ax1, ax2] = plt.subplots(
        nrows=1, ncols=2, figsize=(13, 5), constrained_layout=True
    )

    ax1.plot(dslist, e_change_list, "o")
    ax1.plot(x, np.exp(const_fit) * x**slope_fit)
    ax1.set_title(r"Difference of energy densities")
    yla = ax1.set_ylabel(r"$|\Delta \frac{\mathrm{d}E}{\mathrm{d}s}|$/norm")
    ax1.set_xlabel(r"$\mathrm{d}s$")

    ax2.plot(log_ds, log_e_change, "o")
    ax2.plot(x_log, slope_fit * x_log + const_fit)
    ax2.set_title(
        r"Difference of energy densities, logarithmic, slope: {}".format(
            round(slope_fit, 4)
        )
    )
    ax2.set_xlabel(r"log(d$s$)")
    ax2.set_ylabel(
        r"$\log{\left( |\Delta \frac{\mathrm{d}E}{\mathrm{d}s}|/\mathrm{norm} \right)}$"
    )

    plt.savefig(
        "{}dE(ds)_lambda={}_gamma={}_nz={}.png".format(
            where_to_save_data, lambdabar, gamma, n_z
        )
    )
    plt.show()


def track_gamma1(slist, R0):
    gamma = np.sqrt(slist**2 + R0**2) / R0
    return gamma


def track_gamma2(phi, r, R0, phi00):
    phi_r0 = phi00 / 2
    index_radius = (np.abs(phi - phi_r0)).argmin()
    radius = r[index_radius]

    gamma = radius / R0
    return gamma


def gamma_test(gamma, ds):

    # finding phi(z) that evolves in s:
    initial = two_bubbles(phi_b, phi_false, alpha, gamma, "one")
    z, phi, n_z, d, r_info, phi00 = initial
    dz = abs(z[1] - z[0])

    r0, r_in, r_out = r_info

    # calculating the number of times the field is evolved
    smax = d * 1.2
    n = int(round((smax) / ds))
    s = np.linspace(0, smax, n + 1)
    ds = s[1] - s[0]

    # simulation
    phiend = evolve(phi, ds, dz, n)

    # finding gammas
    gamma1 = track_gamma1(s, r0)
    gamma2 = np.zeros((n + 1))
    for i in range(n + 1):
        phi = phiend[i]
        gamma2[i] = track_gamma2(phi, z, r0, phi00)

    to_save = itertools.zip_longest(s, gamma1, gamma2)
    with open(
        "{}gamma_validity_lambda={}_gamma={}.txt".format(
            where_to_save_data, lambdabar, gamma
        ),
        "w",
    ) as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["s", "gamma_theor", "gamma_sim"])
        writer.writerows(to_save)

    fig, [ax1, ax2] = plt.subplots(
        nrows=1, ncols=2, figsize=(9, 4), constrained_layout=True
    )

    ax1.plot(s, gamma2, color="blue", label="with simulated values")
    ax1.plot(s, gamma1, color="red", label="theoretical")
    ax1.set_title(r"Profiles of $\gamma$ in $s$")
    ax1.set_xlabel(r"$s$", fontsize=14)
    yla = ax1.set_ylabel(r"$\gamma$", fontsize=15, labelpad=8)
    yla.set_rotation(0)
    ax1.legend()

    ax2.plot(s, (gamma1 - gamma2) / gamma1)
    ax2.set_title(r"Difference of $\gamma$")
    ax2.set_xlabel(r"$s$", fontsize=14)
    yla2 = ax2.set_ylabel(
        r"$\frac{\Delta \gamma}{\gamma_{\mathrm{theor}}}$", fontsize=16, labelpad=16
    )
    yla2.set_rotation(0)

    plt.savefig(
        "{}gamma_expansion_test_gamma={}_lambdabar={}.png".format(
            where_to_save_data, gamma, lambdabar
        )
    )
    plt.show()


def default_run(ds, name):
    # finding the initial field of the critical bubble + some parameters
    initial = two_bubbles(phi_b, phi_false, alpha, gamma, name)
    z = initial[0]
    phi = initial[1]
    n_z = initial[2]
    d = initial[3]
    r_info = initial[4]
    dz = abs(z[1] - z[0])

    # if dz is smaller than ds, let's make ds smaller
    if dz <= ds:
        ds = dz / 2
    if set_ds:
        ds = ds_hard

    # calculating the number of times the field is evolved
    smax = d * 1.2
    n = int(round((smax) / ds))
    slist = np.linspace(0, smax, n + 1)
    if set_ds:
        slist = np.arange(0, smax + ds, ds)
        n = len(slist)
    ds = slist[1] - slist[0]

    # printing important info
    print("\nparameters:")
    print("gamma     = " + str(gamma))
    print("lambdabar = " + str(lambdabar))
    print("r_0       = " + str(r_info[0]))
    print("r_in      = " + str(r_info[1]))
    print("r_out     = " + str(r_info[2]))
    print("d         = " + str(d))
    print("ds        = " + str(ds))
    print("dz        = " + str(dz))
    print("n_s       = " + str(n))
    print("n_z       = " + str(n_z))
    print("dz        = " + str(dz), flush=True)

    # evolving the field n times
    phiend = evolve(phi, ds, dz, n)

    # calculating the energy density every tenth step
    energy_density = calculate_energy_density(phiend, ds, dz)

    # information to save
    phi_to_save = phiend[::how_often_ds]
    phi_mid = phiend[:, 0]
    info = (lambdabar, gamma, d, ds, n_z, how_often_ds, r_info)
    to_dump = (info, phi_to_save, z, phi_mid, energy_density)

    # pickling the data
    if dz_mult != 1 or set_ds:
        with open(
            "{}values_lambda={}_gamma={}_{}_dztimes{}_ds={}.pickle".format(
                where_to_save_data, lambdabar, gamma, name, dz_mult, ds
            ),
            "wb",
        ) as f:
            pickle.dump(to_dump, f)
    else:
        with open(
            "{}values_lambda={}_gamma={}_{}.pickle".format(
                where_to_save_data, lambdabar, gamma, name
            ),
            "wb",
        ) as f:
            pickle.dump(to_dump, f)

    # plotting the energy density

    plt.imshow(
        energy_density[::-1],
        cmap="RdBu_r",
        extent=[np.amin(z), np.amax(z), 0, len(energy_density[:, 0]) * 10 * ds],
    )
    cbar = plt.colorbar()
    plt.ylabel("s")
    plt.xlabel("z")
    plt.title("energy density")
    plt.show()

    # plotting a 2D plot of the field evolution
    plot_color(z, phiend, ds, n)

    # plotting phi(z) graphs for a video
    if video:
        plot_phi(z, phiend, how_often_ds, n, ds, dz)


# starting the run (or test)
if not test:
    default_run(ds, "half")
else:
    if which_test == "one":
        default_run(ds, "one")
    elif which_test == "energy_derivative_comparison":
        try:
            ds = param[5]
            ds = float(ds)
            ds_set = True
        except:
            ds_set = False
            ds = 0.01
        energy_derivative_test(gamma, ds, ds_set)
    elif which_test == "energy_derivative_convergence":
        energy_derivative_convergence_test(
            lower=0.001, upper=0.02, gap=0.002, gamma=gamma
        )
    elif which_test == "gamma_test":
        try:
            ds = param[5]
            ds = float(ds)
        except:
            ds = 0.01
        gamma_test(gamma, ds)
    else:
        print("Extra argument given that doesn't correspond to existing tests")
