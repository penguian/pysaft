#!/usr/bin/env python
import argparse
import bioarray as ba
import numpy as np
import scipy as sp
import saftstats
import timeit

alphabet = ba.dna_alphabet
alpha = len(alphabet)
alpha_freq = sp.ones(alpha) / alpha

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

args = parse_args()

# Parse input and database sequences and build frequency matrices.

inp_freq, inp_size, inp_id = ba.build_dna_sparse_frequency_matrix(args.input, args.wordsize)
dat_freq, dat_size, dat_id = ba.build_dna_sparse_frequency_matrix(args.database, args.wordsize)

# Calculate d2.
        
#print timeit.timeit("d2vals = inp_mat.T * dat_mat",number=1,setup="from __main__ import inp_mat, dat_mat")

d2_vals = inp_freq.T * dat_freq

# Calculate and print p values.

inp_len = inp_freq.shape[1]
dat_len = dat_freq.shape[1]

context = saftstats.saft_stats_context(args.wordsize, alpha_freq)
""" 
for i in xrange(inp_len):
    print ""
    print inp_id[i], ":"
    for j in xrange(dat_len):
        d2_mean = saftstats.saft_stats_mean(context, inp_size[i], dat_size[i])
        d2_var  = saftstats.saft_stats_var(context, inp_size[i], dat_size[i])
        d2_pval = saftstats.saft_stats_pgamma_m_v(d2_vals[i, j], d2_mean, d2_var)
        if d2_pval < args.pmax:
            print ">", dat_id[j]
            print "Hit: d2 ==", d2_vals[i, j], ", p-value == ", d2_pval
""" 
 
d2_means = np.zeros((inp_len, dat_len))
d2_vars  = np.zeros((inp_len, dat_len))
d2_pvals = np.zeros((inp_len, dat_len))
 
for i in xrange(inp_len):
    for j in xrange(dat_len):
        d2_means[i,j] = saftstats.saft_stats_mean(context, inp_size[i], dat_size[j])
        d2_vars[i,j]  = saftstats.saft_stats_var(context, inp_size[i], dat_size[j])

d2_pvals = saftstats.saft_stats_pgamma_m_v(d2_vals.todense(), d2_means, d2_vars)

for i in xrange(inp_len):
    print ""
    print inp_id[i], ":"
    for j in xrange(dat_len):
        if d2_pvals[i, j] < args.pmax:
            print ">", dat_id[j]
            print "Hit: d2 ==", d2_vals[i, j], ", p-value == ", d2_pvals[i, j]
            
           