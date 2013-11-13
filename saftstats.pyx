"""
 * saftstats.pyx based on satfstats.c
 * Copyright (C) 2008  Sylvain FORET
 * Copyright (C) 2013  Paul LEOPARDI
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
#cython: cdivsion=True
#cython: cdivision_warnings=True

import numpy as np
import scipy.stats
from cython_gsl cimport gsl_cdf_gamma_Q

cdef double sum_freq_pow (object frequencies,
                          unsigned int freq_pow,
                          unsigned int sum_pow):
    cdef double res = 0
    cdef unsigned int length = len(frequencies)
    cdef unsigned int i
    for i in xrange(length):
        res += frequencies[i] ** freq_pow
    res = res ** sum_pow
    return res

"""
 * FIXME There's quite a bit of optimisation and checking for numeric stability
 * that could be done here
"""

cdef class stats_context:
    cdef double p_2_k
    cdef double sum_var_Yu
    cdef double cov_crab
    cdef double cov_diag
    cdef double cov_ac1
    cdef double cov_ac2
    cdef unsigned int word_size
    cdef unsigned int unif

    def __cinit__(self, word_size, letters_frequencies):

        cdef int          n_letters = len(letters_frequencies)
        cdef int          k = word_size
        cdef unsigned int i
        cdef unsigned int j

        def p(freq_pow, sum_pow):
            return sum_freq_pow(letters_frequencies, freq_pow, sum_pow)

        self.word_size = word_size
        self.p_2_k     = p(2, k)
        self.cov_crab  = 0
        self.cov_diag  = 0
        self.cov_ac1   = 0
        self.cov_ac2   = 0
        self.unif      = True

        for i in xrange(1, n_letters):
            if (letters_frequencies[i] != letters_frequencies[0]):
                self.unif = False
                break

        self.sum_var_Yu = p (2, k) - p (2, 2 * k)

        if not self.unif:
            self.cov_crab = (p(3, k) +
                2 * p(2, 2) * p(3, 1) * ((p(3, k - 1) - p(2, 2 * (k - 1))) / (p(3, 1) - p(2, 2))) -
                (2 * k - 1) * p(2, 2 * k))

        if self.word_size == 1:
            return

        self.cov_diag = (p(2, k + 1) * ((1 - p(2, k - 1)) / (1 - p(2, 1))) -
            (k - 1) * p(2, 2 * k)) * 2

        cdef unsigned int nu
        cdef unsigned int ro
        for i in xrange(1, k):
            for j in xrange(i):
                nu = (k - j) / (i - j)
                ro = (k - j) % (i - j)

                self.cov_ac1 += (p(2, 2 * j) * p(2 * nu + 3, ro) * p(2 * nu + 1, i - j - ro) -
                    p(2, 2 * k))
        self.cov_ac1 *= 4

        cdef unsigned int x
        cdef double prod1
        cdef double prod2
        cdef unsigned int t
        for i in xrange(1, k):
            for j in xrange(1, k):
                nu    = k / (i + j)
                ro    = k % (i + j)
                prod1 = 1
                prod2 = 1

                for x in xrange(1, j + 1):
                    t = 1 + 2 * nu
                    if (x <= ro):
                        t += 1
                    if (x + i <= ro):
                        t += 1
                    prod1 *= p(t, 1)
                for x in xrange(1, i + 1):
                    t = 1 + 2 * nu
                    if (x <= ro):
                        t += 1
                    if (x + j <= ro):
                        t += 1
                    prod2 *= p(t, 1)
                self.cov_ac2 += prod1 * prod2
        self.cov_ac2 -= (k - 1) * (k - 1) * p(2, 2 * k)
        self.cov_ac2 *= 2

cpdef double mean (stats_context context,
                   unsigned int  query_size,
                   unsigned int  subject_size):

    cdef double m = query_size
    cdef double n = subject_size
    cdef double mn = m * n
    return mn * context.p_2_k

cpdef double var (stats_context context,
                  unsigned int  query_size,
                  unsigned int  subject_size):

    cdef double m = query_size
    cdef double n = subject_size
    cdef double mn = m * n
    cdef int    k = context.word_size

    cdef double cov_crab = (n + m - 4 * k + 2) * context.cov_crab

    if (context.word_size == 1):
        return mn * (context.sum_var_Yu + cov_crab)

    return mn * (context.sum_var_Yu + cov_crab + context.cov_diag + context.cov_ac1 + context.cov_ac2)

def pgamma_m_v (d2, mean, var):
    scale = var / mean
    shape = mean / scale

    cdef unsigned int nbr_pvals = d2.shape[0]

    result = np.empty(d2.shape)
    cdef unsigned int i
    for i in xrange(nbr_pvals):
        result[i] = gsl_cdf_gamma_Q (d2[i], shape[i], scale[i])
    return result

"""
 * Benjamini and Hochberg method
 * p_values are expected to be already sorted in increasing order
"""
def BH_array (p_values, tot_n_p_values=0):
    cdef int i

    result = np.empty(p_values.shape)
    n_p_values = p_values.shape[0]
    if  tot_n_p_values < n_p_values:
        tot_n_p_values = n_p_values
    for i in xrange(n_p_values - 1, -1, -1):
        result[i] = (p_values[i] * tot_n_p_values) / (i + 1)
        if i < n_p_values - 1:
            if  result[i] > result[i + 1]:
                result[i] = result[i + 1];
    return result
