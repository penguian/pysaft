pysaft
======

Prototype of (potentially faster) version of SAFT using SciPy sparse matrix multiplication

Prerequisites
-------------

Pysaft depends on Python, NumPy, SciPy, Cython, CythonGSL, Gnu Scientific Library and gcc.

Versions and sources used to produce results of 2013-08-19:

* [Python 2.7.3 -- via Kubuntu](http://www.python.org/download/releases/2.7.3/)
* [NumPy 1.6.1 -- via Kubuntu](https://sourceforge.net/projects/numpy/files/NumPy/1.6.1/)
* [SciPy 0.9.0 -- via Kubuntu](https://sourceforge.net/projects/scipy/files/scipy/0.9.0/)
* [Cython 0.19.1](https://pypi.python.org/pypi/Cython/)
* [CythonGSL 0.2.1 - via GitHub](https://github.com/twiecki/CythonGSL)
* [GSL 1.15 -- via Kubuntu](http://mirror.aarnet.edu.au/pub/gnu/gsl/)
* [gcc 4.6.3 -- via Kubuntu](http://mirror.aarnet.edu.au/pub/gnu/gcc/)

To make
-------

Pysaft uses a rudimentary Makefile. Main choices are:

* make
* make clean
