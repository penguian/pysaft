"""
 * saftmpi.pyx
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
from mpi4py import MPI

class Process:
    def __init__(self, args):
        """
        Establish which process MPI thinks this is.
        """
        self.comm_world = MPI.COMM_WORLD
        world_comm = self.comm_world
        self.nbr_rows = args.mpi_rows
        self.nbr_cols = (world_comm.size - 1) // self.nbr_rows
        self.rank = world_comm.rank
        if self.rank != 0:
            self.row = (self.rank - 1) // self.nbr_cols
            self.col = (self.rank - 1) %  self.nbr_cols
        """
        Create a communicator for each process row.
        """
        world_group = world_comm.Get_group()
        if self.rank == 0:
            self.row_comms = []
        cdef unsigned int mpi_row
        for mpi_row in xrange(self.nbr_rows):
            if self.rank == 0 or mpi_row == self.row:
                row_ranks = [0] + range(self.nbr_cols*mpi_row + 1,
                                        self.nbr_cols*(mpi_row + 1)  + 1)
                row_group = world_group.Incl(row_ranks)
                row_comm = world_comm.Create(row_group)
                if self.rank == 0:
                    self.row_comms.append(row_comm)
                else:
                    self.row_comm  = row_comm
            else:
                """
                Open MPI has a bug which produces a segfault if
                MPI.GROUP_EMPTY is used in the call to Incl().
                """
                row_group = world_group.Incl([0]) #MPI.GROUP_EMPTY
                row_comm = world_comm.Create(row_group)

    def gather_send_size(self, size_vec, size):
        """
        Send a vector of sizes to the scribe process to use with Gatherv.
        """
        size_val = np.array(size, dtype=np.int64)
        row_comm = self.row_comm
        row_comm.Gather([size_val, MPI.LONG],
                        [size_vec, MPI.LONG])

    def gather_recv_size(self, size_vec):
        """
        Receive a vector of sizes to use with Gatherv.
        """
        size_val = np.array(0, dtype=np.int64)
        row_comm = self.row_comms[0]
        row_comm.Gather([size_val, MPI.LONG],
                        [size_vec, MPI.LONG])

    def gatherv_send(self, size_vec, value_vec):
        """
        Gather a vector of values into the scribe process,
        with sizes given by size_vec,
        """
        mpi_type =  MPI.__TypeDict__[value_vec.dtype.char]
        mpi_row_comm = self.row_comm
        mpi_row_comm.Gatherv([value_vec, mpi_type],
                             [None, (size_vec, None), mpi_type])

    def gatherv_recv(self, size_vec, value_vec, row):
        """
        Gather a vector of values into the scribe process,
        with sizes given by size_vec,
        """
        mpi_type =  MPI.__TypeDict__[value_vec.dtype.char]
        mpi_row_comm = self.row_comms[row]
        mpi_row_comm.Gatherv([None, mpi_type],
                             [value_vec, (size_vec, None), mpi_type])
