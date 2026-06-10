from scipy.special.cython_special cimport jv

# C versions of maths functions, NB real numbers only
from libc.math cimport sqrt, exp, sin, cos
import cython



# NB: this has to be a cdef, not a cpdef or def, because these add
# extra stuff to the argument list to help python. LowLevelCallable
# does not like these things...

# You can however increase the number of arguments (remember also to
# update test.pxd)
cdef double integrand(int n, double[9] args):
    cdef double u = args[0]
    # coefficients
    cdef double w = args[1]
    cdef double k = args[2]
    cdef double s = args[3]
    cdef double sign = args[4]
    cdef double complex imag = args[5]
    if (imag == -1):
      imag = -1j
    cdef double t_cut = args[6]
    cdef double t_0 = args[7]
    int_num = args[8]

    cdef double complex result = 0.0

    # defining integrals, which one is used for integration depends on
    # the given parameter value
    if (int_num == 1):
      result =imag * (u**2+sign) * (cos(w*s*u) + 1j*sin(w*s*u)) * \
              (jv(0,in_bessel(w,k,s,u,sign))-jv(2,in_bessel(w,k,s,u,sign))) * \
              C1(s*u, t_cut,t_0)
    elif (int_num == 2):
      result = imag * (u**2+sign) * (cos(w*s*u) + 1j*sin(w*s*u)) * \
              (jv(0,in_bessel(w,k,s,u,sign))+jv(2,in_bessel(w,k,s,u,sign))) * \
              C1(u*s, t_cut,t_0)
    elif (int_num == 3):
      result = imag * (cos(w*s*u) + 1j*sin(w*s*u)) * \
              jv(0,in_bessel(w,k,s,u,sign)) * C1(u*s, t_cut,t_0)
    elif (int_num == 4):
      result = imag * sign * (cos(w*s*u) + 1j*sin(w*s*u)) * sqrt(u**2+sign) * \
              jv(1,in_bessel(w,k,s,u,sign)) * C1(u*s, t_cut,t_0)

    return result.real
    # scipy.integrate takes only real numbers,
    # imaginary part is achieved with changing variable imag

# function to plug inside the bessel fucntion
cdef double in_bessel(double w, double k, double s, double u, double sign):
  return w*sqrt(1-k*k)*s*sqrt(u*u+sign)


# the cutoff function
# The decorator stops division by zero checking
@cython.cdivision(True)
cdef double C1(double t, double t_cut, double t_0):
    if (t<t_cut):
        return 1
    else:
        return exp(-(t-t_cut)**2/t_0**2)
