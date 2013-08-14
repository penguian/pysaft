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

inp_freq, inp_size, inp_desc = saftsparse.build_dna_sparse_frequency_matrix(args.input, args.wordsize)
dat_freq, dat_size, dat_desc = saftsparse.build_dna_sparse_frequency_matrix(args.database, args.wordsize)

print "Sequence parse time ==", "{:f}".format( time() - tick )

# Calculate d2.

tick = time()

d2_vals = np.asarray((inp_freq.T * dat_freq).todense())
# print "vals  type ==", type(d2_vals)
# print "vals  shape==", d2_vals.shape

print "Calculate d2   time ==", "{:f}".format( time() - tick )

# Calculate theroretical means and vars.

tick = time()

inp_len = inp_freq.shape[1]
dat_len = dat_freq.shape[1]

context = saftstats.stats_context(args.wordsize, alpha_freq)
 
#d2_means = np.zeros((inp_len, dat_len))
#d2_vars  = np.zeros((inp_len, dat_len))
#d2_pvals = np.zeros((inp_len, dat_len))

d2_means = np.array([[saftstats.mean(context, inp_size[i], dat_size[j]) 
                      for j in xrange(dat_len)] for i in xrange(inp_len)])
d2_vars  = np.array([[saftstats.var(context, inp_size[i], dat_size[j]) 
                      for j in xrange(dat_len)] for i in xrange(inp_len)])

# print "vals  type ==", type(d2_vals)
# print "vals  shape==", d2_vals.shape
# print "means type==",  type(d2_means)
# print "means shape==", d2_means.shape
# print "vars  type==",  type(d2_vars)
# print "vars  shape==", d2_vars.shape
 
#for i in xrange(inp_len):
#    for j in xrange(dat_len):
#        d2_means[i,j] = saftstats.mean(context, inp_size[i], dat_size[j])
#        d2_vars[i,j]  = saftstats.var(context, inp_size[i], dat_size[j])

print "Means and vars time ==", "{:f}".format( time() - tick )

# Calculate p values.

tick = time()

d2_pvals = np.array(saftstats.pgamma_m_v(d2_vals, d2_means, d2_vars))
                         
# d2_pvalsnew = np.array([[saftstats.pgamma_m_v(d2_vals[i, j], 
#                                           d2_means[i, j], 
#                                           d2_vars[i, j])
#                      for j in xrange(dat_len)] for i in xrange(inp_len)])
# print d2_pvals - d2_pvalsnew

# d2_pvals = np.array([[saftstats.pgamma_m_v(d2_vals[i, j], 
#                                            saftstats.mean(context, inp_size[i], dat_size[j]), 
#                                            saftstats.var(context, inp_size[i], dat_size[j]))
#                       for j in xrange(dat_len)] for i in xrange(inp_len)])
# print d2_pvals.shape

print "Calc p-values  time ==", "{:f}".format( time() - tick )

# Print p values.

tick = time()

for i in xrange(inp_len):
    print ""
    print inp_desc[i], ":"
    jrange = [j for j in xrange(dat_len) if d2_pvals[i, j] < args.pmax]
    d2_vals_i  = np.array(d2_vals[i, :])
    d2_pvals_i = np.array(d2_pvals[i, :])
    #print d2_vals_i.shape
    #print d2_pvals_i.shape
    for j in jrange:
        print ">", dat_desc[j]
        # print "Hit: d2 ==", d2_vals[i, j], ", p-value == ", d2_pvals[i, j]
        # print j
        print "Hit: d2 ==", d2_vals_i[j], ", p-value == ", d2_pvals_i[j]

print "Print p-values time ==", "{:f}".format( time() - tick )
