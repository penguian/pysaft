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

import numpy as np
import saftargs
import saftsparse
import saftstats
import saftmpi
from time import time

def print_mpi_rank(rank):
    print "Process", rank, ":",

def print_elapsed_time(rank, message, elapsed_time):
    print_mpi_rank(rank)
    print message, "==", "{:f}".format(elapsed_time)

# Parse arguments.

tick = time()

args = saftargs.parse_args(mpi_args=True)

# Determine alphabet size and letter frequency.

alphabet = saftsparse.dna_alphabet
alpha = len(alphabet)
alpha_freq = np.ones(alpha) / alpha

# Establish which process MPI thinks this is,
# Create a communicator for each process row.

my_process = saftmpi.Process(args)
my_rank = my_process.rank

if my_rank > my_process.nbr_rows * my_process.nbr_cols:
    print_mpi_rank(my_rank)
    print "Warning: process is redundant:",
    print "nbr_rows ==", my_process.nbr_rows,
    print "comm_world.size ==", my_process.comm_world.size

if args.timing and my_rank in {0,1}:
    print_elapsed_time(my_rank, "Communicator   time", time() - tick)

# Parse database sequences and build database frequency matrix.

if args.timing and my_rank in {0,1}:
    tick = time()

if my_rank == 0:
    dat_desc = saftsparse.build_dna_desc_list(args.database)
    dat_len = len(dat_desc)
else:
    dat_freq, dat_size, dat_desc = saftsparse.build_dna_sparse_frequency_matrix(
        args.database,
        args.wordsize,
        my_process.col,
        my_process.nbr_cols,
        getdesc=False)
    dat_len = dat_freq.shape[1]

if args.timing and my_rank in {0,1}:
    print_elapsed_time(my_rank, "Database parse time", time() - tick)

# Get the number of database sequences from each worker process
# in process row 0. This number should be the same for all process rows.

if args.timing and my_rank in {0,1}:
    tick = time()

dat_len_vec = np.zeros(my_process.nbr_cols + 1, dtype=np.int64)

if my_rank != 0:

    # Determine the number of p values within this process
    # to send back to the scribe process.

    nbr_pvals = min(args.showmax, dat_len)

if my_rank == 0:
    dat_len_val = np.array(0, dtype=np.int64)
    my_process.gather_recv_size(dat_len_vec)

    # Zero out entry zero: we don't want to send ourselves unnecessary data.

    dat_len_vec[0] = 0
elif my_process.row == 0:
    dat_len_val = np.array(nbr_pvals, dtype=np.int64)
    my_process.gather_send_size(dat_len_vec, dat_len_val)

if args.timing and my_rank in {0,1}:
    print_elapsed_time(my_rank, "Get nbr p-vals time", time() - tick)

if my_rank != 0:
    if args.timing and my_rank == 1:
        d2_time = 0
        mv_time = 0
        pv_time = 0
        ad_time = 0

    # Process each query, one at a time.

    for inp_freq, inp_size, inp_desc in saftsparse.gen_dna_frequency(
        args.input,
        args.wordsize,
        start=my_process.row,
        step=my_process.nbr_rows,
        getdesc=False):

        # Calculate d2.

        if args.timing and my_rank == 1:
            tick = time()

        d2_vals = np.asarray((inp_freq.T * dat_freq).todense())[0]

        if args.timing and my_rank == 1:
            d2_time += time() - tick

        # Calculate theoretical means and vars.

        if args.timing and my_rank == 1:
            tick = time()

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

        if args.timing and my_rank == 1:
            mv_time += time() - tick

        # Calculate p values.

        if args.timing and my_rank == 1:
            tick = time()

        d2_pvals = saftstats.pgamma_m_v_vector(d2_vals, d2_means, d2_vars)

        if args.timing and my_rank == 1:
            pv_time += time() - tick

        # Sort by p value within this process.

        if args.timing and my_rank == 1:
            tick = time()

        jsorted = np.argsort(d2_pvals)

        # Cut off by number of p values within this process.

        j_vals = jsorted[:nbr_pvals]

        # Define the local to global mapping of column indices to use
        # to send to the scribe process.

        local_to_global = lambda j: my_process.col + j * my_process.nbr_cols

        # Store the global indices resulting from the sorting.

        j_vals_global = np.array(map(local_to_global, j_vals),
                                 dtype=np.int64)

        # Gather the global indices, D2 values and p values into
        # the scribe process.

        my_process.gatherv_send(dat_len_vec, j_vals_global)
        my_process.gatherv_send(dat_len_vec, d2_vals[j_vals])
        my_process.gatherv_send(dat_len_vec, d2_pvals[j_vals])

        if args.timing and my_rank == 1:
            ad_time += time() - tick

    if args.timing and my_rank == 1:
        print_elapsed_time(my_rank, "Calculate d2   time", d2_time)
        print_elapsed_time(my_rank, "Means and vars time", mv_time)
        print_elapsed_time(my_rank, "Calc p-values  time", pv_time)
        print_elapsed_time(my_rank, "Adj p-values   time", ad_time)

else:
    if args.timing:
        tick = time()

    # Initialize the vectors d2_vals for D2 values, and d2_pvals for p values.

    j_vals   = np.empty(dat_len, dtype=np.int64)
    d2_vals  = np.empty(dat_len, dtype=np.double)
    d2_pvals = np.empty(dat_len, dtype=np.double)

    # Print the results of each query, one at a time.

    i = 0
    for inp_desc in saftsparse.gen_dna_desc(args.input):

        # Gather the global indices, D2 values and p values into
        # the scribe process.

        process_row = i % my_process.nbr_rows
        my_process.gatherv_recv(dat_len_vec, j_vals,  process_row)
        my_process.gatherv_recv(dat_len_vec, d2_vals, process_row)
        my_process.gatherv_recv(dat_len_vec, d2_pvals,process_row)

        print "Query:", inp_desc,
        print "program: saftn word size:", args.wordsize

        # Sort, adjust, cutoff and print p values.

        nbr_pvals = np.sum(dat_len_vec[1 : my_process.nbr_cols + 1])
        jsorted = np.argsort(d2_pvals[:nbr_pvals])
        d2_adj_pvals = saftstats.BH_array(d2_pvals[jsorted], dat_len)
        nbr_pvals = min(args.showmax, nbr_pvals)
        jrange = [j for j in xrange(nbr_pvals)
                  if d2_adj_pvals[j] < args.pmax]
        if len(jrange) > 0:
            for j in jrange:
                js = jsorted[j]
                jg = j_vals[js]
                print "  Hit:", dat_desc[jg],
                print "D2:", "{:d}".format(long(d2_vals[js])),
                print "adj.p.val:", "{:11.5e}".format(d2_adj_pvals[j]),
                print "p.val:", "{:11.5e}".format(d2_pvals[js])
        else:
            print "No hit found"
        i += 1

    if args.timing:
        print_elapsed_time(my_rank, "Print p-values time", time() - tick)
