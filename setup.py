from distutils.core import setup
from Cython.Build import cythonize
import os
ext_name = os.environ['ext_name']
source   = os.environ['source_pyx']

setup(
    name = ext_name,
    ext_modules = cythonize(source),
)
