#from distutils.core import setup
#from Cython.Build import cythonize

#setup(
    #name = "saftstats",
    #ext_modules = cythonize('saftstats.pyx'),
#)

from distutils.core import setup
from Cython.Distutils import Extension
from Cython.Distutils import build_ext
import cython_gsl

setup(
    include_dirs = [cython_gsl.get_include()],
    cmdclass = {'build_ext': build_ext},
    ext_modules = [Extension("saftstats",
                             ["saftstats.pyx"],
                             libraries=cython_gsl.get_libraries(),
                             library_dirs=[cython_gsl.get_library_dir()],
                             include_dirs=[cython_gsl.get_cython_include_dir()])]
    )