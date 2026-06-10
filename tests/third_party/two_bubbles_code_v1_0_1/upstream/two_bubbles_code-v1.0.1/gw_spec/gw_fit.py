#!/usr/bin/python
# gw-fit.py
# tool for fitting gw spectra

#
# libraries
#
import sys
import getopt
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib import rc
from scipy.interpolate import interp1d

rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"]})
rc("axes", labelsize=18)
rc("axes", titlesize=18)
rc("xtick", labelsize=12)
rc("ytick", labelsize=12)
rc("legend", fontsize=12)
rc("text", usetex=True)

#
# methods
#


# mass
def mass_broken(params):
    l = params["lambdabar"]
    mbsq = (9 - 8 * l + 3 * np.sqrt(9 - 8 * l)) / 18.0
    return np.sqrt(mbsq)


# mass
def mass_symmetric(params):
    l = params["lambdabar"]
    mssq = 2 * l / 9.0
    return np.sqrt(mssq)


# get mass point
def mass_broken_point(data, params):
    mb = params["d"] * mass_broken(params)
    f = interp1d(data[:, 0], data[:, 1], kind="linear")
    return [mb, f(mb)]


# get mass point
def mass_symmetric_point(data, params):
    mb = params["d"] * mass_symmetric(params)
    f = interp1d(data[:, 0], data[:, 1], kind="linear")
    return [mb, f(mb)]


# cut off for fit
def wd_max(params):
    # return np.sqrt(params["d"] * mass_broken(params))
    return min(
        params["d"] * mass_broken(params),
        params["d"] * mass_symmetric(params),
        10 * np.pi,
    )


# function to fit, Eq (49) in arXiv:2005.13537
def fn_49(k, Omega0, k0, b):
    a = 3
    return (
        Omega0
        * (a + b)
        * np.power(k, a)
        * np.power(k0, b)
        / (a * np.power(k, a + b) + b * np.power(k0, a + b))
    )


# doing fit fn
def fit_49(data):
    i_max = np.argmax(data[:, 1])
    Omega0 = data[i_max, 1]
    k0 = data[i_max, 0]
    guess = [Omega0, k0, 1.5]
    popt, pcov = curve_fit(fn_49, data[:, 0], data[:, 1], p0=guess)
    return popt, pcov


# printing fit params
def print_params(params, cov):
    print("Omega_0 = " + str(params[0]) + " (" + str(np.sqrt(cov[0][0])) + ")")
    print("k_0     = " + str(params[1]) + " (" + str(np.sqrt(cov[1][1])) + ")")
    print("a       = " + str(3.0) + " (" + str(0.0) + ")")
    print("b       = " + str(params[2]) + " (" + str(np.sqrt(cov[2][2])) + ")")


# raw string label
def raw_label(label, value, form):
    return r"{}".format("$" + label + " = " + str.format("{:.3g}", value) + "$")


# from https://stackoverflow.com/questions/29260893/
def scientific_notation(number, sig_fig):
    ret_string = "{0:.{1:d}e}".format(number, sig_fig)
    a, b = ret_string.split("e")
    # remove leading "+" and strip leading zeros
    b = int(b)
    return a + "\\times 10^{" + str(b) + "}"


# raw string label in scientific notation
def raw_sci_label(label, value, figs):
    return r"{}".format("$" + label + " = " + scientific_notation(value, figs) + "$")


# get parameters
def get_dict(f):
    names = np.loadtxt(f, usecols=0, dtype=str)
    values = np.loadtxt(f, usecols=1, dtype=float)
    d = dict(zip(names, values))
    return d


# normalisation of GW spectra
def GW_norm(params):
    l = params["lambdabar"]
    v = (3 + np.sqrt(9 - 8 * l)) / 6.0
    V = l * v**2 / 9.0 - v**3 / 3.0 + v**4 / 4.0
    d = params["d"]
    s = params["ds"] * params["n_s"]
    # O(1) constant in vol just set to a sphere with radius s
    # see KTW between Eqs. (48) and (49), also after (C7)
    vol = (4 * np.pi / 3.0) * s**3
    return (8 * np.pi / 3.0) * d**2 * V**2 * vol


# normalisation of GW spectra
def GW_norm_envelope():
    d = 240.0
    t = 1.2 * d
    # O(1) constant in vol just set to a sphere with radius t
    # see KTW between Eqs. (48) and (49), also after (C7)
    vol = (4 * np.pi / 3.0) * t**3
    return (8 * np.pi / 3.0) * d**2 * vol


# example usage
def help():
    print(sys.argv)
    print("e.g. usage:")
    print("  python gw_fit.py data-file-1 params-file-1 ... data-file-n params-file-n")


#
# main
#


def main(argv):

    # files
    files = ""  # data files to load
    file_env = "gw_results_ktw-envelope-vw1.0-tmax288-d240.dat"

    # options
    do_plot = True  # do plot
    annotate_plot = False  # add annotations to plot
    verbose = False  # print extra bits and bobs
    envelope = False  # plot envelope approximation too

    # plot stuff
    colours = [
        "#029e73",
        "#de8f05",
        "#0173b2",
        "#56b4e9",
        "#ece133",
        "#949494",
        "#fbafe4",
        "#ca9161",
        "#cc78bc",
        "#d55e00",
    ]
    places = [[0.05, 0.3], [0.3, 0.3], [0.05, 0.05], [0.3, 0.05]]
    add_cross = False
    add_plus = False

    # getting argv
    if len(sys.argv) <= 2 or len(sys.argv) % 2 != 1:
        help()
        sys.exit()
    else:
        files = sys.argv[1:]

    # setting up plot
    if do_plot:
        fig, ax = plt.subplots(1, 1)
        [plot_xmin, plot_xmax] = [1, 1e2]
        [plot_ymin, plot_ymax] = [1e-6, 5e-3]
        ax.set_xlim([plot_xmin, plot_xmax])
        ax.set_ylim([plot_ymin, plot_ymax])
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$\omega R_*$")
        ax.set_ylabel(
            r"$\frac{1}{(H_*R_*\Omega_{\mathrm{vac}})^2}\frac{d\Omega_{\mathrm{gw}}}{\ln(\omega)}$"
        )
        ax.legend(loc="lower left")

    # envelope approx
    if envelope:
        # loading envelope data
        data_env = np.loadtxt(file_env, usecols=[0, 1, 2], dtype=float)
        data_env = data_env[np.argsort(data_env[:, 0])]
        data_env = np.unique(data_env, axis=0)
        n_env = len(data_env)
        # normalising envelope data
        data_env[:, 0] = data_env[:, 0] * 240.0
        norm = GW_norm_envelope()
        data_env[:, 1] = data_env[:, 1] / norm
        data_env[:, 2] = data_env[:, 2] / norm
        # fitting data
        fit_params, cov = fit_49(data_env)
        Omega0 = fit_params[0]
        k0 = fit_params[1]
        a = 3
        b = fit_params[2]
        # showing fit parameters
        if verbose:
            print("envelope:")
        else:
            print(
                "...",
                "...",
                1.2,
                Omega0,
                np.sqrt(cov[0][0]),
                k0,
                np.sqrt(cov[1][1]),
                b,
                np.sqrt(cov[2][2]),
            )
        # plotting envelope data
        if do_plot:
            lab = r"{}".format("$\\mathrm{envelope}$")
            # plt.errorbar(data_env[:, 0], data_env[:, 1], yerr=data_env[:, 2], fmt='k-', zorder=4, label=lab)
            ax.plot(
                data_env[:, 0], data_env[:, 1], "k-", linewidth=1, zorder=4, label=lab
            )

    # number of spectra
    ns = int(len(files) / 2)

    for i in range(ns):
        # loading data
        fd = files[2 * i]
        data = np.loadtxt(fd, usecols=[0, 1], dtype=float)
        data = data[np.argsort(data[:, 0])]
        data = np.unique(data, axis=0)
        n = len(data)

        # loading params
        fp = files[2 * i + 1]
        params = get_dict(fp)

        # calculating gamma alt
        s_col = np.sqrt((params["d"] / 2) ** 2 - params["r0"] ** 2)
        gamma_alt = (params["r_out"] - params["r_in"]) / (
            np.sqrt(params["r_out"] ** 2 + s_col**2)
            - np.sqrt(params["r_in"] ** 2 + s_col**2)
        )

        # normalising data
        data[:, 0] = data[:, 0] * params["d"]
        norm = GW_norm(params)
        data[:, 1] = data[:, 1] / norm

        # fitting data
        cut_data = data[data[:, 0] < wd_max(params)]
        fit_params, cov = fit_49(cut_data)
        Omega0 = fit_params[0]
        k0 = fit_params[1]
        a = 3
        b = fit_params[2]

        # showing fit parameters
        if verbose:
            print(fd + ":")
            print_params(fit_params, cov)
        else:
            if i == 0:
                print(
                    "lambdabar",
                    "gamma",
                    "gamma_alt",
                    "ds",
                    "dz",
                    "n_s",
                    "n_z",
                    "how_often_ds",
                    "d",
                    "r_in",
                    "r0",
                    "r_out",
                    "Omega0",
                    "Omega0_err",
                    "k0",
                    "k0_err",
                    "b",
                    "b_err",
                )
            print(
                params["lambdabar"],
                params["gamma"],
                gamma_alt,
                params["ds"] / params["how_often_ds"],
                params["dz"],
                int(params["n_s"] + 0.5),
                int(params["n_z"] + 0.5),
                int(params["how_often_ds"] + 0.5),
                params["d"],
                params["r_in"],
                params["r0"],
                params["r_out"],
                Omega0,
                np.sqrt(cov[0][0]),
                k0,
                np.sqrt(cov[1][1]),
                b,
                np.sqrt(cov[2][2]),
            )

        # constructing fit function
        xmin = np.amin(cut_data[:, 0])
        xmax = np.amax(cut_data[:, 0])
        xdata = xmin * np.power(xmax / xmin, np.linspace(0, 1, num=n))
        linedata = fn_49(xdata, *fit_params)

        if do_plot:
            # plotting data
            lab = r"{}".format(
                "$\\bar{\\lambda} = "
                + str.format("{:.3g}", params["lambdabar"])
                + "$"
                + ", $\\gamma = "
                + str(round(params["gamma"]))
                + "$"
            )
            ax.plot(data[:, 0], data[:, 1], color=colours[i], zorder=2, label=lab)

            # plotting fit
            ax.plot(xdata, linedata, "--", color=colours[i], zorder=1)

            # adding crosses at mass scales
            if params["d"] * mass_broken(params) < plot_xmax:
                mb_point = mass_broken_point(data, params)
                ax.plot(
                    [mb_point[0]],
                    [mb_point[1]],
                    "+",
                    markersize=7,
                    color=colours[i],
                    zorder=3,
                )
                add_plus = True
            if params["d"] * mass_symmetric(params) < plot_xmax:
                ms_point = mass_symmetric_point(data, params)
                ax.plot([ms_point[0]], [ms_point[1]], "x", color=colours[i], zorder=3)
                add_cross = True

            # adding table of data to plot
            if annotate_plot and i < len(places):
                x_frac = places[i][0]
                y_frac = places[i][1]
                ax.annotate(
                    raw_sci_label("\\tilde{\\Omega}", Omega0, 2),
                    xy=(x_frac, y_frac + 0.15),
                    xycoords="axes fraction",
                    color=colours[i],
                )
                ax.annotate(
                    raw_label("\\tilde{\\omega}", k0, "{:.3g}"),
                    xy=(x_frac, y_frac + 0.10),
                    xycoords="axes fraction",
                    color=colours[i],
                )
                ax.annotate(
                    raw_label("a", a, "{:.3g}"),
                    xy=(x_frac, y_frac + 0.05),
                    xycoords="axes fraction",
                    color=colours[i],
                )
                ax.annotate(
                    raw_label("b", b, "{:.3g}"),
                    xy=(x_frac, y_frac),
                    xycoords="axes fraction",
                    color=colours[i],
                )

    # where some data has already been plotted to ax
    if do_plot:
        handles, labels = ax.get_legend_handles_labels()
        if add_plus:
            black_plus = mlines.Line2D(
                [],
                [],
                color="k",
                marker="+",
                markersize=7,
                linestyle="None",
                label=r"{}".format("$M_{\\rm t} R_*$"),
            )
            handles.append(black_plus)
        if add_cross:
            black_cross = mlines.Line2D(
                [],
                [],
                color="k",
                marker="x",
                linestyle="None",
                label=r"{}".format("$M_{\\rm f} R_*$"),
            )
            handles.append(black_cross)
        ax.legend(handles=handles, loc="lower left")

    if do_plot:
        # showing plot
        plt.tight_layout()
        plt.savefig("gw.png", format="png")
        plt.savefig("gw.pdf", format="pdf")
        plt.show()


if __name__ == "__main__":
    main(sys.argv[1:])
