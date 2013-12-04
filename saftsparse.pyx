"""
 * saftsparse.pyx
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
import scipy.sparse as ss
import os
from Bio import SeqIO
from tempfile import mkdtemp

def build_alpha_dict(char* alphabet):
    cdef unsigned int alpha = len(alphabet)
    cdef unsigned int k
    alpha_dict = {}
    for k in xrange(alpha):
        alpha_dict[<char>(alphabet[k])] = k
    return alpha_dict

dna_alphabet = "ACGT"
dna_alpha_dict = build_alpha_dict(dna_alphabet)

cdef int restart_pos(alpha_dict, char* seq, unsigned int word_len, unsigned int start_pos):
    cdef unsigned int seq_len = len(seq)
    cdef unsigned int pos = start_pos
    cdef unsigned int result
    for result in xrange(start_pos, seq_len - word_len):
        for rel_pos in xrange(word_len):
            pos = result + rel_pos
            if seq[pos] not in alpha_dict:
                break
        if seq[pos] in alpha_dict:
            return result
    return -1

cdef unsigned int word_to_state(alpha_dict, char* seq, int start_pos, int end_pos):
    cdef unsigned int alpha = len(alpha_dict)
    cdef unsigned int result = alpha_dict[seq[start_pos]]
    cdef int pos
    for pos in xrange(start_pos + 1, end_pos):
        result = result * alpha + alpha_dict[seq[pos]]
    return result

cdef build_frequency_vector(alpha_dict, char* seq, unsigned int word_len):
    cdef unsigned int alpha = len(alpha_dict)
    cdef unsigned long freq_len = alpha ** word_len
    cdef unsigned long seq_len = len(seq)
    rows = np.empty((seq_len),dtype=np.uint64)
    cdef unsigned long word_nbr = 0
    cdef unsigned long row
    cdef int start_pos = restart_pos(alpha_dict, seq, word_len, 0)
    restarted = True
    cdef unsigned int end_pos = start_pos + word_len
    while (0 <= start_pos) and (end_pos <= seq_len):
        end_char = seq[end_pos - 1]
        if end_char in alpha_dict:
            if restarted:
                row = word_to_state(alpha_dict, seq, start_pos, end_pos)
            else:
                row = (row * alpha + alpha_dict[end_char]) % freq_len
            rows[word_nbr] = row
            restarted = False
            start_pos += 1
            word_nbr += 1
        else:
            start_pos = restart_pos(alpha_dict, seq, word_len, end_pos)
            restarted = True
        end_pos = start_pos + word_len
    rows = rows[: word_nbr]
    cols = np.zeros((word_nbr), dtype=np.intc)
    data = np.ones(word_nbr, dtype=np.uint32)
    frequency = ss.coo_matrix((data,(rows, cols)), shape=(freq_len, 1))
    return frequency, word_nbr

def gen_dna_desc(file_name):
    with open(file_name) as file_handle:
        for rec in SeqIO.parse(file_handle,"fasta"):
            yield rec.description

def build_dna_desc_list(file_name):
    desc_list = []
    for description in gen_dna_desc(file_name):
        desc_list.append(description)
    return desc_list

def gen_dna_frequency(file_name,
                      unsigned int word_len,
                      unsigned int start=0,
                      unsigned int step=1,
                      getdesc=True,
                      masked=True):
    alpha_dict = dna_alpha_dict
    cdef char* seq
    cdef unsigned int recnbr = 0
    with open(file_name) as file_handle:
        for rec in SeqIO.parse(file_handle,"fasta"):
            if recnbr % step == start:
                recseq = str(rec.seq) if masked else str(rec.seq).upper()
                seq = recseq
                frequency, nbr_words = build_frequency_vector(alpha_dict, seq, word_len)
                description = rec.description if getdesc else ""
                yield frequency, nbr_words, description
            recnbr += 1

def build_dna_sparse_frequency_matrix(file_name,
                                      unsigned int word_len,
                                      unsigned int start=0,
                                      unsigned int step=1,
                                      getdesc=True,
                                      masked=True):
    size_list = []
    desc_list = []
    cdef unsigned long array_len = 2
    rows = np.empty((array_len), dtype=np.intc)
    cols = np.empty((array_len), dtype=np.intc)
    data = np.empty((array_len), dtype=np.uint32)
    cdef unsigned long nnz = 0
    cdef unsigned long next_nnz
    cdef int nbr_cols = 0
    for frequency, nbr_words, description in gen_dna_frequency(
            file_name,
            word_len,
            start=start,
            step=step,
            getdesc=getdesc,
            masked=masked):
        size_list.append(nbr_words)
        if getdesc:
            desc_list.append(description)
        next_nnz = nnz + frequency.nnz
        if array_len < next_nnz:
            array_len = (next_nnz * 3) // 2
            rows.resize((array_len))
            cols.resize((array_len))
            data.resize((array_len))
        rows[nnz:next_nnz] = frequency.row
        cols[nnz:next_nnz] = nbr_cols
        data[nnz:next_nnz] = frequency.data
        nnz = next_nnz
        nbr_cols += 1
    shape = (frequency.shape[0], nbr_cols)
    return ss.csr_matrix((data,(rows, cols)), shape=shape), size_list, desc_list

class slice_end:
    def __init__(self, nnz, nbr_cols):
        self.nnz = nnz
        self.nbr_cols = nbr_cols

class sparse_matrix_memmap:
    def __init__(self, data, rows, cols, shape, ends_list):
        self.data = data
        self.rows = rows
        self.cols = cols
        self.shape = shape
        self.ends_list = ends_list
    def __del__(self):
        try:
            os.unlink(self.data.filename)
            os.unlink(self.rows.filename)
            os.unlink(self.cols.filename)
            os.rmdir(os.path.dirname(self.data.filename))
            os.rmdir(os.path.dirname(self.rows.filename))
            os.rmdir(os.path.dirname(self.cols.filename))
        except OSError:
            pass

def build_dna_sparse_frequency_memmap(file_name,
                                      unsigned int word_len,
                                      unsigned int start=0,
                                      unsigned int step=1,
                                      getdesc=True,
                                      masked=True,
                                      dir=".",
                                      memmap_nnz=10000000000,
                                      slice_nnz =  100000000):
    cdef int nbr_cols = 0
    cdef int this_slice_nbr = 0
    cdef int next_slice_nbr
    cdef unsigned long this_nnz = 0
    cdef unsigned long next_nnz
    size_list = []
    desc_list = []
    ends_list = []
    data = np.memmap(os.path.join(mkdtemp(dir=dir), "datmap.dat"),
                     mode="w+",
                     shape=(memmap_nnz),
                     dtype=np.uint32)
    rows = np.memmap(os.path.join(mkdtemp(dir=dir), "rowmap.dat"),
                     mode="w+",
                     shape=(memmap_nnz),
                     dtype=np.intc)
    cols = np.memmap(os.path.join(mkdtemp(dir=dir), "colmap.dat"),
                     mode="w+",
                     shape=(memmap_nnz),
                     dtype=np.intc)
    for frequency, nbr_words, description in gen_dna_frequency(
            file_name,
            word_len,
            start=start,
            step=step,
            getdesc=getdesc,
            masked=masked):
        size_list.append(nbr_words)
        if getdesc:
            desc_list.append(description)
        next_nnz = this_nnz + frequency.nnz
        if next_nnz > memmap_nnz:
            print "build_dna_sparse_frequency_memmap failed"
            return None
        next_slice_nbr = next_nnz // slice_nnz
        if next_slice_nbr != this_slice_nbr:
            ends_list.append(slice_end(this_nnz, nbr_cols))
            this_slice_nbr = next_slice_nbr
        data[this_nnz:next_nnz] = frequency.data
        rows[this_nnz:next_nnz] = frequency.row
        cols[this_nnz:next_nnz] = nbr_cols
        this_nnz = next_nnz
        nbr_cols += 1
    ends_list.append(slice_end(this_nnz, nbr_cols))
    shape = (frequency.shape[0], nbr_cols)
    return sparse_matrix_memmap(data, rows, cols, shape,ends_list), size_list, desc_list

def multiply(lhs, rhs):
    """
    Multiply the transpose of matrix lhs on the right by matrix rhs.
    """
    try:
        return np.asarray((lhs.T * rhs).todense())
    except ValueError:
        data_filename = rhs.data.filename
        data_dtype = rhs.data.dtype
        rows_filename = rhs.rows.filename
        rows_dtype = rhs.rows.dtype
        cols_filename = rhs.cols.filename
        cols_dtype = rhs.cols.dtype
        nbr_rows = rhs.shape[0]
        nbr_cols = rhs.shape[1]
        nbr_result_rows = lhs.shape[1]
        result = np.zeros((nbr_result_rows, nbr_cols), dtype=np.uint64)
        nnz_beg = 0
        col_beg = 0
        for this_slice_end in rhs.ends_list:
            nnz_end = this_slice_end.nnz
            col_end = this_slice_end.nbr_cols
            nnz = nnz_end - nnz_beg

            data = np.memmap(data_filename,
                             mode='r',
                             shape=(nnz),
                             dtype=data_dtype,
                             offset=nnz_beg * data_dtype.itemsize)
            rows = np.memmap(rows_filename,
                             mode='r',
                             shape=(nnz),
                             dtype=rows_dtype,
                             offset=nnz_beg * rows_dtype.itemsize)
            cols = np.memmap(cols_filename,
                             mode='r',
                             shape=(nnz),
                             dtype=cols_dtype,
                             offset=nnz_beg * cols_dtype.itemsize)
            shape = (nbr_rows, col_end - col_beg)

            sparseslice=ss.csr_matrix((data,(rows,cols - col_beg)), shape=shape)
            result[:, col_beg : col_end] = (lhs.T * sparseslice).todense()
            nnz_beg = nnz_end
            col_beg = col_end
        return result
