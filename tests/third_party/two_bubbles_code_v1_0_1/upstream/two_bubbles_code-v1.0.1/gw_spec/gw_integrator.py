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
import u_integrand  # cython file
from datetime import datetime


# This python file has only one function, in which are multiple subfunctions
# The functions for integration are defined below, most importantly k_integral()
# which does the actual work.
def gw_integral(
    current_w_num, n_k, w, z, phi, phi2, ds, path_to_data, cutoff_ratio, print_sub=False
):
    smax = (len(phi) - 1) * ds
    slist = np.linspace(0, smax, len(phi))
    dz = abs(z[1] - z[0])

    # at first: defining all necessary functions

    # Turn the Cython C function into a LowLevelCallable
    integrand = scipy.LowLevelCallable.from_cython(u_integrand, "integrand")

    # writing subintegrals to file (only if specified)
    def write_subintegral_results_to_file(name, int):
        file = open("{}subintegral_results.txt".format(path_to_data), "a")
        file.write(name + "\t" + str(int) + "\n")
        file.close()
        print((name), flush=True)
        print((int), flush=True)

    # this u_max cuts off the integral when C(su)<exp(-49)~10^(-21)
    # it would be exactly exp(-49) without the 1
    def u_max(s, tau_c, tau_0):
        return 1 + (tau_c + 7 * tau_0) / s

    # region 1: r^2<t^2 region (integrals A1-A4)
    def integral_u_quad_region1(u_int_func, s, k, w, i_s, ds):
        sign = -1
        res1_real = scipy.integrate.quad(
            integrand,
            1,
            u_max(s, t_cut, t_0),
            args=(w, k, s, sign, 1, t_cut, t_0, u_int_func),
        )
        res1_imag = scipy.integrate.quad(
            integrand,
            1,
            u_max(s, t_cut, t_0),
            args=(w, k, s, sign, -1, t_cut, t_0, u_int_func),
        )
        res1 = res1_real[0] + res1_imag[0] * 1j
        return res1

    # region 2: r^2>t^2 region (integrals A5-A8)
    def integral_u_quad_region2(u_int_func, s, k, w, i_s, ds):
        sign = 1
        res2_real = scipy.integrate.quad(
            integrand,
            0,
            u_max(s, t_cut, t_0),
            args=(w, k, s, sign, 1, t_cut, t_0, u_int_func),
        )
        res2_imag = scipy.integrate.quad(
            integrand,
            0,
            u_max(s, t_cut, t_0),
            args=(w, k, s, sign, -1, t_cut, t_0, u_int_func),
        )
        res2 = res2_real[0] + res2_imag[0] * 1j
        return res2

    #######################################################################
    # OBS! the code in this section is hardcoded in the cython code: u_integrand.py

    # cutoff function
    # def C1(t):
    #     if (t<t_cut):
    #         return 1
    #     else:
    #         return np.exp(-(t-t_cut)**2/t_0**2)

    # function to plug inside the bessel fucntion
    # def in_bessel(w,k, s, u, sign):
    #     return w*math.sqrt(1-k**2)*s*math.sqrt(u**2+sign)

    # u integrands
    integral_du_xx = 1  # -> lambda k, w, u, s, sign: (u**2+sign)*cmath.exp(w*s*u*1j) * (jv(0,in_bessel(w,k,s,u,sign))-jv(2,in_bessel(w,k,s,u,sign))) * C1(s*u)
    integral_du_yy = 2  # -> lambda k, w, u, s, sign: (u**2+sign)*cmath.exp(w*s*u*1j) * (jv(0,in_bessel(w,k,s,u,sign))+jv(2,in_bessel(w,k,s,u,sign))) * C1(u*s)
    integral_du_zz = 3  # -> lambda k, w, u, s, sign: cmath.exp(w*s*u*1j) * jv(0,in_bessel(w,k,s,u,sign)) * C1(u*s)
    integral_du_xz = 4  # -> lambda k, w, u, s, sign: sign * math.sqrt(u**2+sign)*cmath.exp(w*s*u*1j) * jv(1,in_bessel(w,k,s,u,sign)) * C1(u*s)

    #######################################################################

    # dphi/dz and dphi/ds derivatives for xz z integral
    # region 1: r^2<t^2
    xz_dphi_dz = lambda i_s, ds, i_z, dz: (
        1
        / 2
        / dz
        * (
            phi[i_s, i_z]
            - phi[i_s, i_z - 1]
            + phi[i_s - 1, i_z]
            - phi[i_s - 1, i_z - 1]
        )
    )
    xz_dphi_ds = lambda i_s, ds, i_z, dz: (
        1
        / 2
        / ds
        * (
            phi[i_s, i_z]
            - phi[i_s - 1, i_z]
            + phi[i_s, i_z - 1]
            - phi[i_s - 1, i_z - 1]
        )
    )

    # dphi/dz and dphi/ds derivatives for xz z integral
    # region 2: r^2>t^2
    xz_dphi_dz2 = lambda i_s, ds, i_z, dz: (
        1
        / 2
        / dz
        * (
            phi2[i_s, i_z]
            - phi2[i_s, i_z - 1]
            + phi2[i_s - 1, i_z]
            - phi2[i_s - 1, i_z - 1]
        )
    )
    xz_dphi_ds2 = lambda i_s, ds, i_z, dz: (
        1
        / 2
        / ds
        * (
            phi2[i_s, i_z]
            - phi2[i_s - 1, i_z]
            + phi2[i_s, i_z - 1]
            - phi2[i_s - 1, i_z - 1]
        )
    )

    # z integrands
    # region 1: r^2<t^2
    integral_dz_xandy = lambda k, w, i_z, i_s, zval, s: (
        dz * 2 * np.cos(w * k * zval) * ((phi[i_s, i_z] - phi[i_s - 1, i_z]) / ds) ** 2
    )
    integral_dz_zz = lambda k, w, i_z, i_s, zval, s: (
        dz
        * 2
        * np.cos(w * k * (zval - dz / 2))
        * ((phi[i_s, i_z] - phi[i_s, i_z - 1]) / dz) ** 2
    )
    integral_dz_xz = lambda k, w, i_z, i_s, zval, s: (
        dz
        * 2
        * 1j
        * np.sin(w * k * (zval - dz / 2))
        * xz_dphi_ds(i_s, ds, i_z, dz)
        * xz_dphi_dz(i_s, ds, i_z, dz)
    )

    # z integrands
    # region 2: r^2>t^2
    integral_dz_xandy2 = lambda k, w, i_z, i_s, zval, s: (
        dz
        * 2
        * np.cos(w * k * zval)
        * ((phi2[i_s, i_z] - phi2[i_s - 1, i_z]) / ds) ** 2
    )
    integral_dz_zz2 = lambda k, w, i_z, i_s, zval, s: (
        dz
        * 2
        * np.cos(w * k * (zval - dz / 2))
        * ((phi2[i_s, i_z] - phi2[i_s, i_z - 1]) / dz) ** 2
    )
    integral_dz_xz2 = lambda k, w, i_z, i_s, zval, s: (
        dz
        * 2
        * 1j
        * np.sin(w * k * (zval - dz / 2))
        * xz_dphi_ds2(i_s, ds, i_z, dz)
        * xz_dphi_dz2(i_s, ds, i_z, dz)
    )

    # the integration function
    def k_integral(n_k, print_subintegrals=False):

        klist = np.linspace(0, 1, n_k)
        dk = klist[1] - klist[0]
        intk = np.zeros(n_k)

        if print_subintegrals:
            write_subintegral_results_to_file("\ndk", dk)

        print("w =", w)

        # k integral
        int_k = 0
        for i_k, k in enumerate(klist):
            if k == 1 or k == -1:
                continue

            zmax = z[-1]

            # initializing the s integrals
            int_s_zz = 0
            int_s_xandy = 0
            int_s_xz = 0

            # zz integral
            for i_s, s in enumerate(slist):
                if s == 0:
                    continue

                # zz u integral
                int_u_zz_region1 = integral_u_quad_region1(
                    integral_du_zz, s, k, w, i_s, ds
                )
                int_u_zz_region2 = integral_u_quad_region2(
                    integral_du_zz, s, k, w, i_s, ds
                )

                # zz z integral (calculated at the spots (i_s*ds,i_z*dz-dz/2) due to the derivative)
                int_z_zz_region1 = 0
                int_z_zz_region2 = 0
                for i_z in range(1, len(z)):
                    zval = z[i_z]
                    int_z_zz_region1 += integral_dz_zz(k, w, i_z, i_s, zval, s)
                    int_z_zz_region2 += integral_dz_zz2(k, w, i_z, i_s, zval, s)

                int_zz_u_z_12 = (
                    int_u_zz_region1 * int_z_zz_region1
                    + int_u_zz_region2 * int_z_zz_region2
                )
                if i_s == len(slist) - 1 or i_s == 0:
                    int_s_zz += i_s**2 * ds**3 * int_zz_u_z_12 / 2
                else:
                    int_s_zz += i_s**2 * ds**3 * int_zz_u_z_12

            # xx and yy integrals
            for i_s, s in enumerate(slist):
                if i_s == 0:
                    continue
                s_off = s - ds / 2

                # integral u xx
                int_u_xx_region1 = integral_u_quad_region1(
                    integral_du_xx, s_off, k, w, i_s, ds
                )
                int_u_xx_region2 = integral_u_quad_region2(
                    integral_du_xx, s_off, k, w, i_s, ds
                )

                # integral u yy
                int_u_yy_region1 = integral_u_quad_region1(
                    integral_du_yy, s_off, k, w, i_s, ds
                )
                int_u_yy_region2 = integral_u_quad_region2(
                    integral_du_yy, s_off, k, w, i_s, ds
                )

                # same z integral for xx and yy (calculated at the spots (i_s*ds-ds/2,i_z*dz) due to the derivative)
                int_z_xandy_region1 = 0
                int_z_xandy_region2 = 0
                for i_z in range(0, len(z)):
                    zval = z[i_z]
                    if i_z == 0 or i_z == len(z) - 1:
                        int_z_xandy_region1 += (
                            integral_dz_xandy(k, w, i_z, i_s, zval, s) / 2
                        )
                        int_z_xandy_region2 += (
                            integral_dz_xandy2(k, w, i_z, i_s, zval, s) / 2
                        )
                    else:
                        int_z_xandy_region1 += integral_dz_xandy(
                            k, w, i_z, i_s, zval, s
                        )
                        int_z_xandy_region2 += integral_dz_xandy2(
                            k, w, i_z, i_s, zval, s
                        )

                # int_s_xx += 1/2*s_off**2*ds * int_z_xandy * int_u_xx
                # int_s_yy += 1/2*s_off**2*ds * int_z_xandy * int_u_yy

                int_u_xandy_1 = int_u_xx_region1 * k**2 - int_u_yy_region1
                int_u_xandy_2 = int_u_xx_region2 * k**2 - int_u_yy_region2

                int_s_xandy += (
                    1
                    / 2
                    * s_off**2
                    * ds
                    * (
                        int_u_xandy_1 * int_z_xandy_region1
                        + int_u_xandy_2 * int_z_xandy_region2
                    )
                )

            # integral xz
            for i_s, s in enumerate(slist):
                if i_s == 0:
                    continue

                s_off = s - ds / 2

                # integral u xz
                int_u_xz_region1 = integral_u_quad_region1(
                    integral_du_xz, s_off, k, w, i_s, ds
                )
                int_u_xz_region2 = integral_u_quad_region2(
                    integral_du_xz, s_off, k, w, i_s, ds
                )

                # integral z xz (calculated at the spots (i_s*ds-ds/2,i_z*dz-dz/2) due to the derivative)
                int_z_xz_region1 = 0
                int_z_xz_region2 = 0
                for i_z in range(1, len(z)):
                    zval = z[i_z]
                    int_z_xz_region1 += integral_dz_xz(k, w, i_z, i_s, zval, s)
                    int_z_xz_region2 += integral_dz_xz2(k, w, i_z, i_s, zval, s)

                int_s_xz += (
                    1j
                    * s_off**2
                    * ds
                    * (
                        int_u_xz_region1 * int_z_xz_region1
                        + int_u_xz_region2 * int_z_xz_region2
                    )
                )

            if print_subintegrals:
                write_subintegral_results_to_file("\nk", k)
                write_subintegral_results_to_file("int s zz", int_s_zz)
                write_subintegral_results_to_file("int s x and y", int_s_xandy)
                write_subintegral_results_to_file("int s xz", int_s_xz)
                write_subintegral_results_to_file("int k", intk[i_k])

            # integrand value
            intk[i_k] = (
                (
                    abs(
                        int_s_zz * (1 - k**2)
                        + int_s_xandy
                        - 2 * int_s_xz * k * np.sqrt(1 - k**2)
                    )
                )
                ** 2
                * w**3
                * 2
                * np.pi
            )

            # k is symmetric around 0, multiply by 2 to take negative k values into account
            # trapezian rule
            if i_k == 0 or i_k == len(klist) - 1:
                int_k += intk[i_k] * dk
            else:
                int_k += 2 * intk[i_k] * dk

            time_now = datetime.now()
            print("k =", k)
            print("int k:", intk[i_k])
            frac_complete = (i_k + 1) / n_k
            print("fraction complete: " + str(frac_complete), flush=True)
            print(time_now.strftime("%d/%m/%Y %H:%M:%S"))

        if print_subintegrals:
            write_subintegral_results_to_file("\ndE/d(log_w)", int_k)

        print("dk =", dk)
        print("dE/d(log_w) = " + str(int_k))

        return w, int_k, klist, intk

    # now that functions have been defined, they will be used

    # setting cutoff parameters
    tau = smax
    t_cut = cutoff_ratio * tau
    t_0 = (tau - t_cut) / 4

    # now the actual running of the code
    results = k_integral(n_k, print_sub)

    return results
