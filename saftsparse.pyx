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
#from Bio.SeqIO import FastaIO
from time import time

dna_alphabet = "ACGT"

def SimpleFastaParser(handle):
    """Generator function to iterator over Fasta records (as string tuples).

    For each record a tuple of two strings is returned, the FASTA title
    line (without the leading '>' character), and the sequence (with any
    whitespace removed). The title line is not divided up into an
    identifier (the first word) and comment or description.

    >>> for values in SimpleFastaParser(open("Fasta/dups.fasta")):
    ...     print values
    ('alpha', 'ACGTA')
    ('beta', 'CGTC')
    ('gamma', 'CCGCC')
    ('alpha (again - this is a duplicate entry to test the indexing code)', 'ACGTA')
    ('delta', 'CGCGC')

    """
    #Skip any text before the first record (e.g. blank lines, comments)
    while True:
        line = handle.readline()
        if line == "":
            return  # Premature end of file, or just empty?
        if line[0] == ">":
            break

    while True:
        if line[0] != ">":
            raise ValueError(
                "Records in Fasta files should start with '>' character")
        title = line[1:].rstrip()
        lines = []
        line = handle.readline()
        while True:
            if not line:
                break
            if line[0] == ">":
                break
            lines.append(line.rstrip())
            line = handle.readline()

        #Remove trailing whitespace, and any internal spaces
        #(and any embedded \r which are possible in mangled files
        #when not opened in universal read lines mode)
        yield title, "".join(lines).replace(" ", "").replace("\r", "")

        if not line:
            return  # StopIteration

    assert False, "Should not reach this line"
     
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
    frequency = np.zeros((freq_len, 1))
    # frequency = ss.dok_matrix((alpha ** word_len, 1))
    cdef unsigned int seq_len = len(seq)
    cdef unsigned long nbr_words = 0
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
            restarted = False
            frequency[row, 0] += 1
            start_pos += 1
            nbr_words += 1
        else:   
            start_pos = restart_pos(alpha_dict, seq, word_len, end_pos)
            restarted = True
        end_pos = start_pos + word_len
    return frequency, nbr_words

def build_dna_frequency_lists(file_name, unsigned int word_len, masked=True):
    cdef char* alphabet = dna_alphabet
    cdef unsigned int alpha = len(alphabet)
    alpha_dict = {}
    for k in xrange(alpha):
        alpha_dict[<char>(alphabet[k])] = k
    file_handle = open(file_name)
    freq_list = []
    size_list = []
    id_list = []
    cdef char* seq
    #for rec in SeqIO.parse(file_handle,"fasta"):
    #for recid, recseq in FastaIO.SimpleFastaParser(file_handle):
    for recid, recseq in SimpleFastaParser(file_handle):
        #recseq = str(rec.seq) if masked else str(rec.seq).upper()
        reqseq = recseq if masked else recseq.upper()
        seq = recseq
        frequency, nbr_words = build_frequency_vector(alpha_dict, seq, word_len)
        freq_list.append(ss.coo_matrix(frequency))
        # freq_list.append(frequency)
        size_list.append(nbr_words)
        #id_list.append(rec.id)
        id_list.append(recid)
    file_handle.close()
    return freq_list, size_list, id_list

def build_dna_sparse_frequency_matrix(file_name, word_len, masked=True):
    freq_list, size_list, id_list = build_dna_frequency_lists(file_name, word_len, masked)
    return ss.hstack(freq_list ,format="csr"), size_list, id_list
