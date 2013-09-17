from distutils.core import setup
from Cython.Build import cythonize

setup(
    name = "MPIsaftsparse",
    ext_modules = cythonize('MPIsaftsparse.pyx'),
)
