# for setup, run following:
# python3 setup.py build_ext --inplace


from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=cythonize(
        Extension("u_integrand", sources=["u_integrand.pyx"], include_dirs=["./"]),
        annotate=True,
    )
)
