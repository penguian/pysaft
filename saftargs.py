"""
 * saftargs.py
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

def parse_args(
        description="Prototype SAFT using SciPy sparse matrix multiplication",
        mpi_args=False):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--input",
                        action="store",
                        required=True,
                        help="Path to the input file")
    parser.add_argument("--database",
                        action="store",
                        required=True,
                        help="Path to the output file")
    parser.add_argument("--wordsize",
                        action="store",
                        type=int,   default=7,
                        help="Word size (7)")
    parser.add_argument("--showmax",
                        action="store",
                        type=int,
                        default=50,
                        help="Maximum number of results to show (50)")
    parser.add_argument("--pmax",
                        action="store",
                        type=float,
                        default=0.05,
                        help="Show results with a p-value smaller than this (0.05)")
    parser.add_argument("--timing",
                        action="store_true",
                        default=False,
                        help="Time key processing steps (False)")
    if mpi_args:
        parser.add_argument("--mpi_rows",
                            action="store",
                            type=int,
                            default=1,
                            help="Number of MPI grid process rows for the input file")
    return parser.parse_args()
