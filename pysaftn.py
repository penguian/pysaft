#!/usr/bin/env python
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
                        help="path to the input file")
    parser.add_argument("--database", action="store", required=True, 
                        help="path to the output file")
    parser.add_argument("--wordsize", action="store", type=int,   default=7,     
                        help="word size")
    parser.add_argument("--pmax",     action="store", type=float, default=0.05,  
                        help="show results with a p-value smaller than this")
    return parser.parse_args()

# Parse arguments.

tick = time()

args = parse_args()

print "Argument parse time ==", "{:f}".format( time() - tick )

# Parse input and database sequences and build frequency matrices.

tick = time()

inp_freq, inp_size, inp_id = saftsparse.build_dna_sparse_frequency_matrix(args.input, args.wordsize)
dat_freq, dat_size, dat_id = saftsparse.build_dna_sparse_frequency_matrix(args.database, args.wordsize)

print "Sequence parse time ==", "{:f}".format( time() - tick )

# Calculate d2.

tick = time()

d2_vals = inp_freq.T * dat_freq

print "Calculate d2   time ==", "{:f}".format( time() - tick )

# Calculate theroretical means and vars.

tick = time()

inp_len = inp_freq.shape[1]
dat_len = dat_freq.shape[1]

context = saftstats.stats_context(args.wordsize, alpha_freq)
 
d2_means = np.zeros((inp_len, dat_len))
d2_vars  = np.zeros((inp_len, dat_len))
d2_pvals = np.zeros((inp_len, dat_len))
 
for i in xrange(inp_len):
    for j in xrange(dat_len):
        d2_means[i,j] = saftstats.mean(context, inp_size[i], dat_size[j])
        d2_vars[i,j]  = saftstats.var(context, inp_size[i], dat_size[j])

print "Means and vars time ==", "{:f}".format( time() - tick )

# Calculate p values.

tick = time()

d2_pvals = saftstats.pgamma_m_v(d2_vals.todense(), d2_means, d2_vars)

print "Calc p-values  time ==", "{:f}".format( time() - tick )

# Print p values.

tick = time()

for i in xrange(inp_len):
    print ""
    print inp_id[i], ":"
    for j in xrange(dat_len):
        if d2_pvals[i, j] < args.pmax:
            print ">", dat_id[j]
            print "Hit: d2 ==", d2_vals[i, j], ", p-value == ", d2_pvals[i, j]

print "Print p-values time ==", "{:f}".format( time() - tick )
