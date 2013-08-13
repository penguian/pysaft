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
from Bio import SeqIO
from time import time

dna_alphabet = "ACGT"
     
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
    cols = np.zeros((word_nbr),dtype=np.uint8)
    data = np.ones(word_nbr)
    frequency = ss.coo_matrix((data,(rows, cols)), shape=(freq_len, 1))
    return frequency, word_nbr

def build_dna_frequency_lists(file_name, unsigned int word_len, masked=True):
    cdef char* alphabet = dna_alphabet
    cdef unsigned int alpha = len(alphabet)
    alpha_dict = {}
    for k in xrange(alpha):
        alpha_dict[<char>(alphabet[k])] = k
    freq_list = []
    size_list = []
    desc_list = []
    cdef char* seq
    with open(file_name) as file_handle:
        for rec in SeqIO.parse(file_handle,"fasta"):
            recseq = str(rec.seq) if masked else str(rec.seq).upper()
            seq = recseq
            frequency, nbr_words = build_frequency_vector(alpha_dict, seq, word_len)
            freq_list.append(frequency)
            size_list.append(nbr_words)
            desc_list.append(rec.description)
    return freq_list, size_list, desc_list

def build_dna_sparse_frequency_matrix(file_name, word_len, masked=True):
    freq_list, size_list, desc_list = build_dna_frequency_lists(file_name, word_len, masked)
    return ss.hstack(freq_list ,format="csr"), size_list, desc_list
