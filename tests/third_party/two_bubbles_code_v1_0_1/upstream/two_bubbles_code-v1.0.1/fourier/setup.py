# python3 setup.py build_ext --inplace


from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=cythonize(
        Extension("fou", sources=["fou.pyx"], include_dirs=["./"]), annotate=True
    )
)
