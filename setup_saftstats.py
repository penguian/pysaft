from distutils.core import setup
from Cython.Build import cythonize

setup(
    name = "saftstats",
    ext_modules = cythonize('saftstats.pyx'),
)