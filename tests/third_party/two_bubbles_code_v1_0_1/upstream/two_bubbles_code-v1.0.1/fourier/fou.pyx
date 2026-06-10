from libc.math cimport sqrt, sin, cos, pi


cdef double sinc(double x):
    if (abs(x)<10**(-3)):
        return 1.0 - x**2/6.0 + x**4/120.0 - x**6/5040.0 + x**8/362880.0
    else:
        return sin(x)/x


cdef double basis_fn_s(int i, int j, int N):
    return sqrt(2.0/N) * (pi*i)/N * sinc(i*j*pi/N)


cdef double weight_fn_s(int j):
    return j**2


cdef double basis_fn_z(int i, int j, int N):
    return sqrt(2.0/(N-1)) * cos(i*j*pi/(N-1))


# the fourier transformation: a combination of sinc transform in s and type-I cos transform in z
def ft(ftphi, phi, N_s, N_z):
    for i_s in range(N_s):
        for i_z in range(N_z):
            for j_s in range(N_s):
                for j_z in range(N_z):
                    if j_z == 0 or j_z == N_z-1:
                        ftphi[i_s][i_z] += 0.5 * weight_fn_s(j_s+1) * phi[j_s][j_z] * basis_fn_s(i_s+1, j_s+1, N_s+1) * basis_fn_z(i_z, j_z, N_z)
                    else:
                        ftphi[i_s][i_z] += weight_fn_s(j_s+1) * phi[j_s][j_z] * basis_fn_s(i_s+1, j_s+1, N_s+1) * basis_fn_z(i_z, j_z, N_z)


# the inverse fourier transformation: weight -> 1 and i<->j within basis functions
def invft(ftphi, phi, N_s, N_z):
    for i_s in range(N_s):
        for i_z in range(N_z):
            for j_s in range(N_s):
                for j_z in range(N_z):
                    if j_z == 0 or j_z == N_z-1:
                        ftphi[i_s][i_z] += 0.5 * phi[j_s][j_z] * basis_fn_s(j_s+1, i_s+1, N_s+1) * basis_fn_z(j_z, i_z, N_z)
                    else:
                        ftphi[i_s][i_z] += phi[j_s][j_z] * basis_fn_s(j_s+1, i_s+1, N_s+1) * basis_fn_z(j_z, i_z, N_z)
