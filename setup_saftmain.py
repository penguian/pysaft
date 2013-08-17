from distutils.core import setup
from Cython.Build import cythonize

setup(
    name = "saftmain",
    ext_modules = cythonize('saftmain.pyx'),
)