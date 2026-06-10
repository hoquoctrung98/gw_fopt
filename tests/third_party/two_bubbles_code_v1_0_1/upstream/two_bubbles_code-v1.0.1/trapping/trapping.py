# another look at trapping
import numpy as np
import matplotlib.pyplot as plt
import pickle
import math
import os
import pandas as pd
from matplotlib import cm
from matplotlib.colors import ListedColormap


# if there's more lambdabars or gammas, add them here
lambdalist = [0.01, 0.07, 0.1, 0.18, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.84, 0.9, 0.95]
gammalist = range(1, 20)


for indexi, lam in enumerate(lambdalist):
    if indexi == 0:
        print(
            "lambdabar",
            "gamma",
            "ds",
            "ds_save",
            "dz",
            "n_s",
            "n_z",
            "d",
            "r_in",
            "r0",
            "r_out",
            "trapping_fraction",
        )
    for indexj, gam in enumerate(gammalist):
        # open a file if it exists, if not -> value=0 aka no data
        try:
            with open(
                "values_lambda={}_gamma={}_half.pickle".format(lam, gam), "rb"
            ) as f:
                things = pickle.load(f, encoding="latin1")
            info = things[0]
            phi = things[1]
            z = things[2]
            phi_mi = things[3]
            lambdabar, gamma, d, ds, n_z, how_often_ds, r_info = info
            r0, r_in, r_out = r_info
            dz = z[1] - z[0]
        except:
            continue
        ds_save = ds * how_often_ds
        s_max = (len(phi) - 1) * ds_save
        s_list = np.linspace(0, s_max, len(phi))

        phi_max = (3 - np.sqrt(9 - 8 * lam)) / 6.0
        phi_max_list = np.zeros_like(phi_mi) + phi_max

        # the collision time
        s_col_0 = np.sqrt((d / 2) ** 2 - r0**2)
        i_s_max = 1
        for i_s in range(1, len(phi_mi) - 1):
            if (
                s_list[i_s] >= s_col_0
                and phi_mi[i_s] > phi_mi[i_s - 1]
                and phi_mi[i_s] > phi_mi[i_s + 1]
                and phi_mi[i_s] > phi_max
            ):
                i_s_max = i_s
                break
        s_col = s_list[i_s_max]

        i_collision = 0
        count_trap = 0
        count_post_collision = 0

        for index in range(len(phi_mi)):
            if s_list[index] > s_col:
                count_post_collision += 1
                if phi_mi[index] < phi_max:
                    count_trap += 1
                if i_collision == 0:
                    i_collision = index

        trapping_fraction = float(count_trap) / count_post_collision
        print(
            lambdabar,
            gamma,
            ds,
            ds_save,
            dz,
            len(phi),
            len(phi[0]),
            d,
            r_in,
            r0,
            r_out,
            trapping_fraction,
        )

        # plt.plot(s_list, phi_mi, 'k-')
        # plt.plot(s_list, phi_max_list, 'b--')
        # plt.axvline(x=s_list[i_collision], color='r')
        # plt.show()
