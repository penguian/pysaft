from distutils.core import setup
from Cython.Build import cythonize

setup(
    name = "saftsparse",
    ext_modules = cythonize('saftsparse.pyx'),
)