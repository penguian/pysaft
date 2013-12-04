"""
 * saftmain.pyx
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

import argparse
import numpy as np
import saftargs
import saftsparse
import saftstage
import saftstats
from time import time

def print_elapsed_time(message, elapsed_time):
    print message, "==", "{:f}".format(elapsed_time)

# Parse arguments.

tick = time()

args = saftargs.parse_args()

if args.timing:
    print_elapsed_time("Argument parse time", time() - tick)

# Determine alphabet size and letter frequency.

alphabet, alpha, alpha_freq = saftstage.setup_dna_alphabet()

# Parse database sequences and build database frequency matrix.

if args.timing:
    tick = time()

if args.memmap:
    dat_freq, dat_size, dat_desc = saftsparse.build_dna_sparse_frequency_memmap(
        args.database,
        args.wordsize)
else:
    dat_freq, dat_size, dat_desc = saftsparse.build_dna_sparse_frequency_matrix(
        args.database,
        args.wordsize)

dat_len = dat_freq.shape[1]

if args.timing:
    print_elapsed_time("Database parse time", time() - tick)

cdef unsigned int i
cdef unsigned int j
cdef unsigned int js

context = saftstats.stats_context(args.wordsize, alpha_freq)

if args.timing:
    qp_time = 0
    d2_time = 0
    mv_time = 0
    pv_time = 0
    pp_time = 0
    tick = time()

if args.memmap:
    inp_freq, inp_size, inp_desc = saftsparse.build_dna_sparse_frequency_matrix(
        args.input,
        args.wordsize)

    inp_len = inp_freq.shape[1]

    if args.timing:
        qp_time += time() - tick

        # Calculate d2.

        if args.timing:
            tick = time()

        d2_vals = saftstage.calculate_d2_statistic(inp_freq, dat_freq)

        if args.timing:
            d2_time += time() - tick

        for i in xrange(inp_len):

            # Calculate theroretical means and vars.

            if args.timing:
                tick = time()

            d2_means, d2_vars = saftstage.calculate_means_vars(
                args,
                context,
                inp_freq[:, i],
                inp_size[i],
                dat_freq,
                dat_size)

            if args.timing:
                mv_time += time() - tick

            # Calculate p values.

            if args.timing:
                tick = time()

            d2_pvals = saftstats.pgamma_m_v(d2_vals[i, :], d2_means, d2_vars)

            if args.timing:
                pv_time += time() - tick

            # Print p values.

            if args.timing:
                tick = time()

            saftstage.print_query_results_d2(
                args,
                inp_desc[i],
                dat_desc,
                d2_vals[i, :],
                d2_pvals,
                dat_len,
                dat_len),

            if args.timing:
                pp_time += time() - tick

        del dat_freq
else:
    for inp_freq, inp_size, inp_desc in saftsparse.gen_dna_frequency(
        args.input,
        args.wordsize):

        if args.timing:
            qp_time += time() - tick

        # Calculate d2.

        if args.timing:
            tick = time()

        d2_vals = saftstage.calculate_d2_statistic(inp_freq, dat_freq)[0,:]

        if args.timing:
            d2_time += time() - tick

        # Calculate theroretical means and vars.

        if args.timing:
            tick = time()

        inp_len = inp_freq.shape[1]

        d2_means, d2_vars = saftstage.calculate_means_vars(
            args,
            context,
            inp_freq,
            inp_size,
            dat_freq,
            dat_size)

        if args.timing:
            mv_time += time() - tick

        # Calculate p values.

        if args.timing:
            tick = time()

        d2_pvals = saftstats.pgamma_m_v(d2_vals, d2_means, d2_vars)

        if args.timing:
            pv_time += time() - tick

        # Print p values.

        if args.timing:
            tick = time()

        saftstage.print_query_results_d2(
            args,
            inp_desc,
            dat_desc,
            d2_vals,
            d2_pvals,
            dat_len,
            dat_len),

        if args.timing:
            pp_time += time() - tick

        if args.timing:
            tick = time()

if args.timing:
    qp_time += time() - tick
    print_elapsed_time("Query parse    time", qp_time)
    print_elapsed_time("Calculate d2   time", d2_time)
    print_elapsed_time("Means and vars time", mv_time)
    print_elapsed_time("Calc p-values  time", pv_time)
    print_elapsed_time("Print p-values time", pp_time)
