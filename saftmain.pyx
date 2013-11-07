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

alphabet = saftsparse.dna_alphabet
alpha = len(alphabet)
alpha_freq = np.ones(alpha) / alpha

# Parse database sequences and build database frequency matrix.

if args.timing:
    tick = time()

dat_freq, dat_size, dat_desc = saftsparse.build_dna_sparse_frequency_matrix(
    args.database,
    args.wordsize)
dat_len = dat_freq.shape[1]

if args.timing:
    print_elapsed_time("Database parse time", time() - tick)

cdef unsigned int j
cdef unsigned int js

if args.timing:
    qp_time = 0
    d2_time = 0
    mv_time = 0
    pv_time = 0
    pp_time = 0
    tick = time()

for inp_freq, inp_size, inp_desc in saftsparse.gen_dna_frequency(
    args.input,
    args.wordsize):

    if args.timing:
        qp_time += time() - tick

    # Calculate d2.

    if args.timing:
        tick = time()

    d2_vals = np.asarray((inp_freq.T * dat_freq).todense())[0]

    if args.timing:
        d2_time += time() - tick

    # Calculate theroretical means and vars.

    if args.timing:
        tick = time()

    inp_len = inp_freq.shape[1]

    context = saftstats.stats_context(args.wordsize, alpha_freq)

    d2_means = np.array([
        saftstats.mean(context,
                       inp_size + args.wordsize - 1,
                       dat_size[j] + args.wordsize - 1)
        for j in xrange(dat_len)])
    d2_vars  = np.array([
        saftstats.var(context,
                      inp_size + args.wordsize - 1,
                      dat_size[j] + args.wordsize - 1)
        for j in xrange(dat_len)])

    if args.timing:
        mv_time += time() - tick

    # Calculate p values.

    if args.timing:
        tick = time()

    d2_pvals = saftstats.pgamma_m_v_vector(d2_vals, d2_means, d2_vars)

    if args.timing:
        pv_time += time() - tick

    # Print p values.

    if args.timing:
        tick = time()

    print "Query:", inp_desc, "program: saftn word size:", args.wordsize
    jsorted = np.argsort(d2_pvals)
    d2_adj_pvals = saftstats.BH_array(d2_pvals[jsorted])
    nbr_pvals = min(args.showmax, d2_adj_pvals.shape[0])
    jrange = [j for j in xrange(nbr_pvals)
              if d2_adj_pvals[j] < args.pmax]
    if len(jrange) > 0:
        for j in jrange:
            js = jsorted[j]
            print "  Hit:", dat_desc[js],
            print "D2:", "{:d}".format(long(d2_vals[js])),
            print "adj.p.val:", "{:11.5e}".format(d2_adj_pvals[j]),
            print "p.val:", "{:11.5e}".format(d2_pvals[js])
    else:
        print "No hit found"

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
