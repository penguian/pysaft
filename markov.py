import sys
import numpy as np
from Bio import SeqIO
from time import time

def restart_pos(alphabet, seq, word_len, start_pos):
    str_len = len(seq)
    for result in xrange(start_pos, str_len - word_len + 1):
        for rel_pos in xrange(word_len):
            pos = result + rel_pos
            if seq[pos] not in alphabet:
                break
        if seq[pos] in alphabet:
            return result
    return -1
    
def word_to_state(alpha_dict, word_str):
    alpha = len(alpha_dict)
    result = alpha_dict[word_str[0]]
    for pos in xrange(1, len(word_str)):
        result = result * alpha + alpha_dict[word_str[pos]]
    return result    
        
def build_array(alphabet, seq, omega):
    alpha_dict = dict(zip(alphabet,range(len(alphabet))))
    alpha = len(alphabet)
    result = np.zeros((alpha ** omega, alpha))
    str_len = len(seq)
    word_len = omega + 1
    pos = restart_pos(alphabet, seq, word_len, 0)
    end_pos = pos + omega
    while (0 <= pos) and (end_pos < str_len):
       if seq[end_pos] in alphabet:
          row = word_to_state(alpha_dict, seq[pos : end_pos]) if pos < end_pos else 0
          col = word_to_state(alpha_dict, seq[end_pos])
          result[row, col] += 1
          pos += 1
       else:   
          pos = restart_pos(alphabet, seq, word_len, end_pos + 1)
       end_pos = pos + omega
    return result

def normalize_array(arr):
    result = np.zeros(arr.shape)
    nbr_rows = result.shape[0]
    nbr_cols = result.shape[1]
    for row in xrange(nbr_rows):
        sum_of_row = sum(arr[row,:])
        if sum_of_row == 0:
            result[row,:] = np.ones((1,nbr_cols)) / nbr_cols
        else:               
            result[row,:] = arr[row,:] / sum_of_row
    return result
    
def print_dna_markov_arrays(file_name, max_omega, masked=True):
    alphabet = "ACGT"
    file_handle = open(file_name)
    np.set_printoptions(threshold=np.nan)
    for rec in SeqIO.parse(file_handle,"fasta"):
        seq = str(rec.seq) if masked else str(rec.seq).upper() 
        print rec.description, ":"
        sys.stdout.flush()
        for omega in xrange(0, max_omega + 1):
            tick = time()
            print "Order ", omega
            arr = build_array(alphabet, seq, omega)
            mat = normalize_array(arr)
            print mat
            print ""
            print "Elapsed: ", time() - tick
            print ""
            sys.stdout.flush()
    file_handle.close()        