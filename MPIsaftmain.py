"""
 * MPIsaftmain.py
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
import MPIsaftsparse as saftsparse
import saftstats
from mpi4py import MPI
from time import time

alphabet = saftsparse.dna_alphabet
alpha = len(alphabet)
alpha_freq = np.ones(alpha) / alpha

def parse_args():
    parser = argparse.ArgumentParser(
        description="Prototype parallel SAFT using SciPy sparse matrix multiplication")
    parser.add_argument("--mpi_rows", action="store", type=int,   default=1,
                        help="Number of MPI grid process rows for the input file")
    parser.add_argument("--input",      action="store", required=True,
                        help="Path to the input file")
    parser.add_argument("--database",   action="store", required=True,
                        help="Path to the output file")
    parser.add_argument("--wordsize",   action="store", type=int,   default=7,
                        help="Word size (7)")
    parser.add_argument("--showmax",    action="store", type=int,   default=50,
                        help="Maximum number of results to show (50)")
    parser.add_argument("--pmax",       action="store", type=float, default=0.05,
                        help="Show results with a p-value smaller than this (0.05)")
    parser.add_argument("--timing",     action="store_true",        default=False,
                        help="Time key processing steps (False)")
    return parser.parse_args()

# Parse arguments.

tick = time()

args = parse_args()

# Establish which process MPI thinks this is.

comm = MPI.COMM_WORLD

mpi_nbr_rows = args.mpi_rows
mpi_nbr_cols = (comm.size - 1) // mpi_nbr_rows
my_mpi_rank = comm.rank
if my_mpi_rank != 0:
    my_mpi_row = (my_mpi_rank - 1) // mpi_nbr_cols
    my_mpi_col = (my_mpi_rank - 1) %  mpi_nbr_cols
    #print "I am worker", my_mpi_rank, my_mpi_row, my_mpi_col
#else:
    #print "I am scribe."

if args.timing and my_mpi_rank == 0:
    print "Argument parse time ==", "{:f}".format( time() - tick )

# Parse input and database sequences and build frequency matrices.

if args.timing and my_mpi_rank == 1:
    tick = time()

if my_mpi_rank == 0:
    inp_desc = saftsparse.build_dna_desc_list(args.input)
    inp_len = len(inp_desc)
    dat_desc = saftsparse.build_dna_desc_list(args.database)
    dat_len = len(dat_desc)
else:
    inp_freq, inp_size, inp_desc = saftsparse.build_dna_sparse_frequency_matrix(
        args.input,    args.wordsize, my_mpi_row, mpi_nbr_rows, desc=True)
    inp_len = inp_freq.shape[1]

    dat_freq, dat_size, dat_desc = saftsparse.build_dna_sparse_frequency_matrix(
        args.database, args.wordsize, my_mpi_col, mpi_nbr_cols, desc=True)
    dat_len = dat_freq.shape[1]

if args.timing and my_mpi_rank == 1:
    print "Sequence parse time ==", "{:f}".format( time() - tick )

if my_mpi_rank != 0:

    # Calculate d2.

    if args.timing and my_mpi_rank == 1:
        tick = time()
    d2_vals = np.asarray((inp_freq.T * dat_freq).todense())

    if args.timing and my_mpi_rank == 1:
        print "Calculate d2   time ==", "{:f}".format( time() - tick )

    # Calculate theroretical means and vars.

    if args.timing and my_mpi_rank == 1:
        tick = time()

    context = saftstats.stats_context(args.wordsize, alpha_freq)

    d2_means = np.array([[saftstats.mean(context, inp_size[i] + args.wordsize - 1, dat_size[j] + args.wordsize - 1)
                          for j in xrange(dat_len)] for i in xrange(inp_len)])
    d2_vars  = np.array([[saftstats.var(context, inp_size[i] + args.wordsize - 1, dat_size[j] + args.wordsize - 1)
                          for j in xrange(dat_len)] for i in xrange(inp_len)])

    if args.timing and my_mpi_rank == 1:
        print "Means and vars time ==", "{:f}".format( time() - tick )

    # Calculate p values.

    if args.timing and my_mpi_rank == 1:
        tick = time()

    d2_pvals = saftstats.pgamma_m_v(d2_vals, d2_means, d2_vars)

    if args.timing and my_mpi_rank == 1:
        print "Calc p-values  time ==", "{:f}".format( time() - tick )

# Create a communicator for each process row

world_group = comm.Get_group()
row_groups = []
row_comms = []
for mpi_row in xrange(mpi_nbr_rows):
    if my_mpi_rank == 0 or mpi_row == my_mpi_row:
        ranks = [0] + range(mpi_row * mpi_nbr_cols + 1, (mpi_row + 1) * mpi_nbr_cols + 1)
        #if my_mpi_rank == 0:
            #print ranks
        mpi_row_group = world_group.Incl(ranks)
        mpi_row_comm = comm.Create(mpi_row_group)
        if my_mpi_rank == 0:
            row_groups.append(mpi_row_group)
            row_comms.append(mpi_row_comm)
        else:
            my_row_group = mpi_row_group
            my_row_comm  = mpi_row_comm
    else:
        mpi_row_group = world_group.Incl([0]) #MPI.GROUP_EMPTY
        # Open MPI has a bug which produces a segfault if
        # MPI.GROUP_EMPTY is used in the call to Incl().
        mpi_row_comm = comm.Create(mpi_row_group)

# Get database lengths across the first row.

dat_len_vec = np.zeros(mpi_nbr_cols + 1, dtype=np.int64)
dat_len_val = np.array(dat_len, dtype=np.int64)
if my_mpi_rank == 0:
    mpi_row_comm = row_comms[0]
    #print "Scribe gathering", i, i % mpi_nbr_rows
    mpi_row_comm.Gather([dat_len_val, MPI.LONG], [dat_len_vec, MPI.LONG])
    # Zero out entry zero - we don't want to send ourselves unnecessary data.
    dat_len_vec[0] = 0
elif my_mpi_row == 0:
    mpi_row_comm = my_row_comm
    #print "Row", my_mpi_row, "gathering"
    mpi_row_comm.Gather([dat_len_val, MPI.LONG], [dat_len_vec, MPI.LONG])

# Print p values.

if args.timing and my_mpi_rank == 0:
    tick = time()


if my_mpi_rank == 0:
    # Initialize the vectors d2_vals_i for D2 values, and d2_pvals_i for p values.
    d2_vals_i  = np.empty(dat_len, dtype=np.double)
    d2_pvals_i = np.empty(dat_len, dtype=np.double)

for i in xrange(inp_len):

    if my_mpi_rank == 0:
        print "Query:", inp_desc[i], "program: saftn word size:", args.wordsize

    if my_mpi_rank != 0:
        # Initialize the vectors d2_vals_i for D2 values, and d2_pvals_i for p values.
        d2_vals_i  = d2_vals[i, :]
        d2_pvals_i = d2_pvals[i, :]

    # Gather the D2 values and p values into the scribe process.
    if my_mpi_rank == 0:
        mpi_row_comm = row_comms[i % mpi_nbr_rows]
        mpi_row_comm.Gatherv([None, MPI.DOUBLE], [d2_vals_i,  (dat_len_vec, None), MPI.DOUBLE])
        mpi_row_comm.Gatherv([None, MPI.DOUBLE], [d2_pvals_i, (dat_len_vec, None), MPI.DOUBLE])
    else:
        mpi_row_comm = my_row_comm
        mpi_row_comm.Gatherv([d2_vals_i,  MPI.DOUBLE], [None, (dat_len_vec, None), MPI.DOUBLE])
        mpi_row_comm.Gatherv([d2_pvals_i, MPI.DOUBLE], [None, (dat_len_vec, None), MPI.DOUBLE])

    # TODO: The order of rhe D2 and p values has been disturbed by the gather,
    # so the old code with argsort and dat_desc[js] will need to be changed.
    if my_mpi_rank == 0:
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

if args.timing and my_mpi_rank == 0:
    print "Print p-values time ==", "{:f}".format( time() - tick )

#if my_mpi_rank == 0:
    #print "I am scribe."
#else:
    #print "I am worker", my_mpi_rank, my_mpi_row, my_mpi_col
