"""
@author: Quoc Trung Ho <qho@sissa.it>
"""
import numpy as np

def derivative(f, x0, dx = 1e-5, order = 2):
    if order == 2:
        return (f(x0 + dx) - f(x0 - dx)) / (2.*dx)
    if order == 4:
        return (-f(x0 + 2*dx)/12 + 2*f(x0 + dx)/3 - 2*f(x0 - dx)/3 + f(x0 - 2*dx)/12) / dx

def derivative_arr(f_list, dx):
    diff_arr = np.zeros(len(f_list))
    diff_arr[1:-1] = (f_list[2:] - f_list[0:-2]) / (2.*dx)
    diff_arr[0] = (-3.*f_list[0] + 4.*f_list[1] - f_list[2]) / dx
    diff_arr[-1] = (3.*f_list[-1] - 4.*f_list[-2] + f_list[-3]) / dx
    return diff_arr
