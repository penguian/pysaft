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

from __future__ import print_function
import argparse
import numpy as np
import saftsparse
import saftstats
import itertools
from mpi4py import MPI
from time import time

alphabet = saftsparse.dna_alphabet
alpha = len(alphabet)
alpha_freq = np.ones(alpha) / alpha

def parse_args():
    parser = argparse.ArgumentParser(
        description="Prototype parallel SAFT using SciPy sparse matrix multiplication")
    parser.add_argument("--input",      action="store", required=True,
                        help="Path to the input file")
    parser.add_argument("--database",   action="store", required=True,
                        help="Path to the output file")
    parser.add_argument("--mpi_rows", action="store", type=int,   default=1,
                        help="Number of MPI grid process rows for the input file")
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

if args.timing and my_mpi_rank in {0,1}:
    print("Process", my_mpi_rank, ":", "Argument parse time ==", "{:f}".format( time() - tick ))

# Parse database sequences and build frequency matrix.

if args.timing and my_mpi_rank in {0,1}:
    tick = time()

if my_mpi_rank == 0:
    dat_desc = saftsparse.build_dna_desc_list(args.database)
    dat_len = len(dat_desc)
else:
    dat_freq, dat_size, dat_desc = saftsparse.build_dna_sparse_frequency_matrix(
        args.database,
        args.wordsize,
        my_mpi_col,
        mpi_nbr_cols,
        getdesc=False)
    dat_len = dat_freq.shape[1]

if args.timing and my_mpi_rank in {0,1}:
    print("Process", my_mpi_rank, ":", "Database parse time ==", "{:f}".format( time() - tick ))

# Create a communicator for each process row.

if args.timing and my_mpi_rank in {0,1}:
    tick = time()

world_group = comm.Get_group()
row_groups = []
row_comms = []
for mpi_row in xrange(mpi_nbr_rows):
    if my_mpi_rank == 0 or mpi_row == my_mpi_row:
        ranks = [0] + range(mpi_row * mpi_nbr_cols + 1, (mpi_row + 1) * mpi_nbr_cols + 1)
        mpi_row_group = world_group.Incl(ranks)
        mpi_row_comm = comm.Create(mpi_row_group)
        if my_mpi_rank == 0:
            row_groups.append(mpi_row_group)
            row_comms.append(mpi_row_comm)
        else:
            my_row_group = mpi_row_group
            my_row_comm  = mpi_row_comm
    else:
        # Open MPI has a bug which produces a segfault if
        # MPI.GROUP_EMPTY is used in the call to Incl().

        mpi_row_group = world_group.Incl([0]) #MPI.GROUP_EMPTY
        mpi_row_comm = comm.Create(mpi_row_group)

if args.timing and my_mpi_rank in {0,1}:
    print("Process", my_mpi_rank, ":", "Communicator   time ==", "{:f}".format( time() - tick ))

# Get the number of database sequences from each worker process in process row 0.
# This number should be the same for all process rows.

if args.timing and my_mpi_rank in {0,1}:
    tick = time()

dat_len_vec = np.zeros(mpi_nbr_cols + 1, dtype=np.int64)

if my_mpi_rank != 0:

    # Determine the number of p values within this process to send back to the scribe process.

    nbr_pvals = min(args.showmax, dat_len)

if my_mpi_rank == 0:
    dat_len_val = np.array(0, dtype=np.int64)
    mpi_row_comm = row_comms[0]
    mpi_row_comm.Gather([dat_len_val, MPI.LONG], [dat_len_vec, MPI.LONG])

    # Zero out entry zero - we don't want to send ourselves unnecessary data.

    dat_len_vec[0] = 0
elif my_mpi_row == 0:
    dat_len_val = np.array(nbr_pvals, dtype=np.int64)
    mpi_row_comm = my_row_comm
    mpi_row_comm.Gather([dat_len_val, MPI.LONG], [dat_len_vec, MPI.LONG])

if args.timing and my_mpi_rank in {0,1}:
    print("Process", my_mpi_rank, ":", "Get nbr seqs time   ==", "{:f}".format( time() - tick ))

if my_mpi_rank != 0:
    if args.timing and my_mpi_rank == 1:
        d2_time = 0
        mv_time = 0
        pv_time = 0
        ad_time = 0

    # Process each query, one at a time.

    for inp_freq, inp_size, inp_desc in saftsparse.gen_dna_frequency(
        args.input,
        args.wordsize,
        start=my_mpi_row,
        step=mpi_nbr_rows,
        getdesc=False):

        # Calculate d2.

        if args.timing and my_mpi_rank == 1:
            tick = time()

        d2_vals = np.asarray((inp_freq.T * dat_freq).todense())[0]

        if args.timing and my_mpi_rank == 1:
            d2_time += time() - tick

        # Calculate theoretical means and vars.

        if args.timing and my_mpi_rank == 1:
            tick = time()

        context = saftstats.stats_context(args.wordsize, alpha_freq)

        d2_means = np.array([
            saftstats.mean(context, inp_size + args.wordsize - 1, dat_size[j] + args.wordsize - 1)
            for j in xrange(dat_len)])
        d2_vars  = np.array([
            saftstats.var(context, inp_size + args.wordsize - 1, dat_size[j] + args.wordsize - 1)
            for j in xrange(dat_len)])

        if args.timing and my_mpi_rank == 1:
            mv_time += time() - tick

        # Calculate p values.

        if args.timing and my_mpi_rank == 1:
            tick = time()

        d2_pvals = saftstats.pgamma_m_v_vector(d2_vals, d2_means, d2_vars)

        if args.timing and my_mpi_rank == 1:
            pv_time += time() - tick

        if args.timing and my_mpi_rank == 1:
            tick = time()

        # Sort by p value within this process.

        jsorted = np.argsort(d2_pvals)

        # Cut off by number of p values within this process.

        j_vals = jsorted[:nbr_pvals]

        # Define the local to global mapping of column indices to use to send to the scribe process.

        local_to_global = lambda j: my_mpi_col + j * mpi_nbr_cols

        # Store the global indices resulting from the sorting.

        j_vals_global = np.array(map(local_to_global, j_vals), dtype=np.int64)

        # Gather the global indices, D2 values and p values into the scribe process.

        mpi_row_comm = my_row_comm
        mpi_row_comm.Gatherv([j_vals_global,    MPI.LONG],     [None, (dat_len_vec, None), MPI.LONG])
        mpi_row_comm.Gatherv([d2_vals[j_vals],  MPI.UNSIGNED], [None, (dat_len_vec, None), MPI.UNSIGNED])
        mpi_row_comm.Gatherv([d2_pvals[j_vals], MPI.DOUBLE],   [None, (dat_len_vec, None), MPI.DOUBLE])

        if args.timing and my_mpi_rank == 1:
            ad_time += time() - tick

    if args.timing and my_mpi_rank == 1:
        print("Process", my_mpi_rank, ":", "Calculate d2   time ==", "{:f}".format( d2_time ))
        print("Process", my_mpi_rank, ":", "Means and vars time ==", "{:f}".format( mv_time ))
        print("Process", my_mpi_rank, ":", "Calc p-values  time ==", "{:f}".format( pv_time ))
        print("Process", my_mpi_rank, ":", "Adj p-values   time ==", "{:f}".format( ad_time ))

else:
    # Print p values.

    if args.timing:
        tick = time()

    # Initialize the vectors d2_vals for D2 values, and d2_pvals for p values.

    j_vals   = np.empty(dat_len, dtype=np.int64)
    d2_vals  = np.empty(dat_len, dtype=np.double)
    d2_pvals = np.empty(dat_len, dtype=np.double)

    # Print the results of each query, one at a time.

    i = 0
    for inp_desc in saftsparse.gen_dna_desc(args.input):

        print("Query:", inp_desc, "program: saftn word size:", args.wordsize)

        # Gather the global indices, D2 values and p values into the scribe process.

        mpi_row_comm = row_comms[i % mpi_nbr_rows]
        mpi_row_comm.Gatherv([None, MPI.LONG],     [j_vals,   (dat_len_vec, None), MPI.LONG])
        mpi_row_comm.Gatherv([None, MPI.UNSIGNED], [d2_vals,  (dat_len_vec, None), MPI.UNSIGNED])
        mpi_row_comm.Gatherv([None, MPI.DOUBLE],   [d2_pvals, (dat_len_vec, None), MPI.DOUBLE])

        # Sort, adjust, cutoff and print p values.

        nbr_pvals = np.sum(dat_len_vec[1 : mpi_nbr_cols + 1])
        jsorted = np.argsort(d2_pvals[:nbr_pvals])
        d2_adj_pvals = saftstats.BH_array(d2_pvals[jsorted], dat_len)
        nbr_pvals = min(args.showmax, nbr_pvals)
        jrange = [j for j in xrange(nbr_pvals) if d2_adj_pvals[j] < args.pmax]
        if len(jrange) > 0:
            for j in jrange:
                js = jsorted[j]
                jg = j_vals[js]
                print("  Hit:", dat_desc[jg],
                      "D2:", "{:d}".format(long(d2_vals[js])),
                      "adj.p.val:", "{:11.5e}".format(d2_adj_pvals[j]),
                      "p.val:", "{:11.5e}".format(d2_pvals[js]))
        else:
            print("No hit found")
        i += 1

    if args.timing:
        print("Process", my_mpi_rank, ":", "Print p-values time ==", "{:f}".format( time() - tick ))
