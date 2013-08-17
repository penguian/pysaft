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
import saftsparse
import saftstats
from time import time

alphabet = saftsparse.dna_alphabet
alpha = len(alphabet)
alpha_freq = np.ones(alpha) / alpha

def parse_args():
    parser = argparse.ArgumentParser(
        description="Prototype faster SAFT using SciPy sparse matrix multiplication")
    parser.add_argument("--input",    action="store", required=True, 
                        help="Path to the input file")
    parser.add_argument("--database", action="store", required=True, 
                        help="Path to the output file")
    parser.add_argument("--wordsize", action="store", type=int,   default=7,     
                        help="Word size (7)")
    parser.add_argument("--showmax",  action="store", type=int,   default=50,  
                        help="Maximum number of results to show (50)")
    parser.add_argument("--pmax",     action="store", type=float, default=0.05,  
                        help="Show results with a p-value smaller than this (0.05)")
    parser.add_argument("--timing",   action="store_true",        default=False,
                        help="Time key processing steps (False)")
    return parser.parse_args()

# Parse arguments.

tick = time()

args = parse_args()

if args.timing:
    print "Argument parse time ==", "{:f}".format( time() - tick )

# Parse input and database sequences and build frequency matrices.

if args.timing:
    tick = time()

inp_freq, inp_size, inp_desc = saftsparse.build_dna_sparse_frequency_matrix(args.input, args.wordsize)
dat_freq, dat_size, dat_desc = saftsparse.build_dna_sparse_frequency_matrix(args.database, args.wordsize)

if args.timing:
    print "Sequence parse time ==", "{:f}".format( time() - tick )

# Calculate d2.

if args.timing:
    tick = time()

d2_vals = np.asarray((inp_freq.T * dat_freq).todense())

if args.timing:
    print "Calculate d2   time ==", "{:f}".format( time() - tick )

# Calculate theroretical means and vars.

if args.timing:
    tick = time()

inp_len = inp_freq.shape[1]
dat_len = dat_freq.shape[1]

context = saftstats.stats_context(args.wordsize, alpha_freq)

cdef unsigned int i
cdef unsigned int j

d2_means = np.array([[saftstats.mean(context, inp_size[i] + args.wordsize - 1, dat_size[j] + args.wordsize - 1) 
                      for j in xrange(dat_len)] for i in xrange(inp_len)])
d2_vars  = np.array([[saftstats.var(context, inp_size[i] + args.wordsize - 1, dat_size[j] + args.wordsize - 1) 
                      for j in xrange(dat_len)] for i in xrange(inp_len)])

if args.timing:
    print "Means and vars time ==", "{:f}".format( time() - tick )

# Calculate p values.

if args.timing:
    tick = time()

d2_pvals = saftstats.pgamma_m_v(d2_vals, d2_means, d2_vars)

if args.timing:
    print "Calc p-values  time ==", "{:f}".format( time() - tick )

# Print p values.

if args.timing:
    tick = time()

cdef unsigned int js

for i in xrange(inp_len):
    print "Query:", inp_desc[i], "program: saftn word size:", args.wordsize
    d2_vals_i  = d2_vals[i, :]
    d2_pvals_i = d2_pvals[i, :]
    jsorted = np.argsort(d2_pvals_i)
    d2_adj_pvals_i = saftstats.BH_array(d2_pvals_i[jsorted])
    nbr_pvals = min(args.showmax, d2_adj_pvals_i.shape[0])
    jrange = [j for j in xrange(nbr_pvals) if d2_adj_pvals_i[j] < args.pmax]
    if len(jrange) > 0:
        for j in jrange:
            js = jsorted[j]
            print "  Hit:", dat_desc[js], "D2:", "{:d}".format(long(d2_vals_i[js])), "adj.p.val:", "{:11.5e}".format(d2_adj_pvals_i[j]), "p.val:", "{:11.5e}".format(d2_pvals_i[js])
    else:
        print "No hit found"

if args.timing:
    print "Print p-values time ==", "{:f}".format( time() - tick )
