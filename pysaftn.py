#!/usr/bin/env python
import bioarray as ba
import scipy as sp
import argparse

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

def saft_stats_sum_freq_pow (f, l, freq_pow, sum_pow):
    res = 0;
    for i in xrange(l):
        res += f[i] ** freq_pow
    res = res ** sum_pow
    return res

alphabet = ba.dna_alphabet
alpha = len(alphabet)
alpha_freq = sp.ones(alpha) / alpha

def p(freq_pow, sum_pow):
    return saft_stats_sum_freq_pow(alpha_freq, alpha, freq_pow, sum_pow)

args = parse_args()

inp_mat = ba.build_dna_sparse_frequency_matrix(args.input, args.wordsize)
dat_mat = ba.build_dna_sparse_frequency_matrix(args.database, args.wordsize)
d2vals = inp_mat.T * dat_mat

print d2vals.shape
print d2vals.nnz
