import sys
import numpy as np
import scipy.sparse as ss
import scipy.sparse.linalg as ssl
from Bio import SeqIO
from time import time

dna_alphabet = "ACGT"

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

def suffix_matches_prefix(alpha, omega, lhs_state, rhs_state):
    suffix = lhs_state % (alpha ** (omega-1))
    prefix = rhs_state // alpha
    return suffix == prefix

def is_markov_shape(alpha, omega, arr, verbose=True):
    nbr_rows = arr.shape[0]
    nbr_cols = arr.shape[1]
    
    if nbr_rows != alpha ** omega:
        if verbose: print "Row size does not match:", nbr_rows, alpha ** omega
        return False
    if nbr_cols != alpha:
        if verbose: print "Col size does not match:", nbr_cols, alpha
        return False
    return True

def big_markov_matrix(alpha, omega, arr):
    nbr_rows = arr.shape[0]
    
    if not is_markov_shape(alpha, omega, arr):
        return None
    matrix_dict = ss.dok_matrix((nbr_rows, nbr_rows))
    for i in xrange(nbr_rows):
        for j in xrange(nbr_rows):
            if suffix_matches_prefix(alpha, omega, i, j):
                matrix_dict[i, j] = arr[i, j % alpha]
    return ss.csr_matrix(matrix_dict)
 
def big_markov_mult(alpha, omega, arr, vec):
    nbr_rows = arr.shape[0]
    
    if not is_markov_shape(alpha, omega, arr):
        return None
    if nbr_rows != vec.shape[0]:
        print "Row size does not match vector:", nbr_rows, vec.shape[1]
        return None
        
    result = np.zeros(vec.shape)
    for j in xrange(nbr_rows):
        result[j] = 0
        for i in xrange(nbr_rows):
            if suffix_matches_prefix(alpha, omega, i, j):
                result[j] += vec[i] * arr[i, j % alpha]
    return result            

build_big_markov_mult = lambda alpha, omega, arr: lambda vec: big_markov_mult(alpha, omega, arr, vec)

def build_stationary_markov_vector(alpha, omega, arr, tol=1.0e-10):
    nbr_rows = alpha ** omega
    mv = build_big_markov_mult(alpha, omega, arr)
    if mv == None:
        return None
    lop = ssl.LinearOperator((nbr_rows, nbr_rows), matvec=mv)
    L, V = ssl.eigs(lop, 1)
    evalue = L.real[0]
    if abs(L[0].real - 1.0) > tol or abs(L[0].imag) > tol:
        print "Error: Eigenvalue is not 1.0: ", L[0]
        return None
    if np.linalg.norm(V.imag) > tol:
        print "Error: Eigenvector is complex: ", V
    V /= np.sum(V)   
    return V.real   

def build_stationary_spectrum(alpha, omega, arr, word_len, tol=1.0e-10):
    if word_len < omega:
        print "Error: Not implemented for word_len < omega: word_len ==", word_len, "omega ==", omega
        return None
    nbr_markov_rows = alpha ** omega
    nbr_spectrum_rows = alpha ** word_len
    nbr_spectrum_per_markov = alpha ** (word_len - omega)
    phi = build_stationary_markov_vector(alpha, omega, arr, tol)
    spectrum = np.zeros((nbr_markov_rows, nbr_spectrum_per_markov))
    for markov_state in xrange(nbr_markov_rows):
        for spectrum_col in xrange(nbr_spectrum_per_markov):
            remainder = spectrum_col
            omega_mer = markov_state
            state_prob = float(phi[markov_state])
            for pos in xrange(word_len - omega):
                remainder_base = alpha ** (word_len - omega - pos - 1)
                letter = remainder // remainder_base
                remainder %= remainder_base
                state_prob *= arr[omega_mer, letter]
                omega_mer = (omega_mer % (alpha ** (omega - 1))) * alpha + letter
            spectrum[markov_state, spectrum_col] = state_prob
    return np.reshape(spectrum, (-1, alpha))
 
def build_frequency_array(alphabet, seq, word_len):
    alpha_dict = dict(zip(alphabet,range(len(alphabet))))
    alpha = len(alphabet)
    omega = word_len - 1
    result = np.zeros((alpha ** omega, alpha))
    str_len = len(seq)
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

def normalize_array_as_markov(arr):
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

def normalize_array_as_spectrum(arr):
    result = arr / float(np.sum(arr))
    return result

def print_array_as_is(arr, description):
    print arr
    
def print_array_as_mma(arr, description):
    nbr_rows = arr.shape[0]
    nbr_cols = arr.shape[1]
    print "%%MatrixMarket matrix array real general"
    print "%"
    print "%", description
    print "%"
    print nbr_rows, nbr_cols
    for col in xrange(nbr_cols):
        for row in xrange(nbr_rows):
            print arr[row, col]

def print_array_as_mmc(arr, description):
    nbr_rows = arr.shape[0]
    nbr_cols = arr.shape[1]
    nbr_entries = nbr_rows * nbr_cols
    print "%%MatrixMarket matrix coordinate real general"
    print "%"
    print "%", description
    print "%"
    print nbr_rows, nbr_cols, nbr_entries
    for col in xrange(nbr_cols):
        for row in xrange(nbr_rows):
            print row + 1, col + 1, arr[row, col]

def print_seq_normalized_frequency_array(alphabet, seq, word_len, description, 
                                         normalize_array_func, print_array_func):
    arr = build_frequency_array(alphabet, seq, word_len)
    mat = normalize_array_func(arr)
    print_array_func(mat, description)

def print_seq_markov_array_as_is(alphabet, seq, omega, description):
    print_seq_normalized_frequency_array(alphabet, seq, omega + 1, description, 
                                         normalize_array_as_markov, print_array_as_is)

def print_seq_markov_array_as_mma(alphabet, seq, omega, description):
    print_seq_normalized_frequency_array(alphabet, seq, omega + 1, description, 
                                         normalize_array_as_markov, print_array_as_mma)

def print_seq_markov_array_as_mmc(alphabet, seq, omega, description):
    print_seq_normalized_frequency_array(alphabet, seq, omega + 1, description, 
                                         normalize_array_as_markov, print_array_as_mmc)

def print_seq_spectrum_array_as_mma(alphabet, seq, omega, description):
    print_seq_normalized_frequency_array(alphabet, seq, omega + 1, description, 
                                         normalize_array_as_spectrum, print_array_as_mma)

def print_stationary_spectrum(alphabet, seq, omega, word_len, description,
                              print_array_func):
    arr = build_frequency_array(alphabet, seq, omega + 1)
    mat = normalize_array_as_markov(arr)
    alpha = len(alphabet)
    spec = build_stationary_spectrum(alpha, omega, mat, word_len)
    header_line = description + ": Markov order " + str(omega) + ", word length " + str(word_len)
    print_array_func(spec, header_line)

def print_stationary_spectrum_as_mma(alphabet, seq, omega, word_len, description):
    print_stationary_spectrum(alphabet, seq, omega, word_len, description,
                              print_array_as_mma)

def build_dna_frequency_array(file_name, word_len, masked=True):
    alphabet = dna_alphabet
    file_handle = open(file_name)
    rec = SeqIO.parse(file_handle,"fasta").next()
    seq = str(rec.seq) if masked else str(rec.seq).upper()
    arr = build_frequency_array(alphabet, seq, word_len)
    file_handle.close()
    return arr
 
def print_dna_arrays(file_name, omega_range, print_seq_array_func, masked=True, verbose=True, timed=True):
    alphabet = dna_alphabet
    file_handle = open(file_name)
    np.set_printoptions(threshold=np.nan)
    for rec in SeqIO.parse(file_handle,"fasta"):
        seq = str(rec.seq) if masked else str(rec.seq).upper()
        if verbose: 
            print rec.description, ":"
            sys.stdout.flush()
        for omega in omega_range:
            if timed:
                tick = time()
            if verbose:
                print "Order ", omega
            print_seq_array_func(alphabet, seq, omega, rec.description)
            if timed:
                print ""
                print "Elapsed: ", time() - tick
                print ""
            sys.stdout.flush()
    file_handle.close()
    
def print_dna_markov_arrays(file_name, max_omega, 
                            masked=True, verbose=True, timed=True):
    print_dna_arrays(file_name, range(max_omega + 1),
                     print_seq_markov_array_as_is, 
                     masked, verbose, timed)
    
def print_dna_markov_array_as_mma(file_name, omega, 
                                  masked=True, verbose=True, timed=True):
    print_dna_arrays(file_name, [omega], 
                     print_seq_markov_array_as_mma, 
                     masked, verbose, timed)
    
def print_dna_markov_array_as_mmc(file_name, omega, 
                                  masked=True, verbose=True, timed=True):
    print_dna_arrays(file_name, [omega], 
                     print_seq_markov_array_as_mmc, 
                     masked, verbose, timed)
    
def print_dna_spectrum_array_as_mma(file_name, word_len, 
                                    masked=True, verbose=False, timed=False):
    print_dna_arrays(file_name, [word_len - 1], 
                     print_seq_spectrum_array_as_mma, 
                     masked, verbose, timed)
    
def print_dna_stationary_spectrum_as_mma(file_name, omega, word_len, 
                                         masked=True, verbose=False, timed=False):
    alphabet = dna_alphabet
    file_handle = open(file_name)
    np.set_printoptions(threshold=np.nan)
    rec = SeqIO.parse(file_handle,"fasta").next()
    seq = str(rec.seq) if masked else str(rec.seq).upper()
    if verbose: 
        print rec.description, ":"
        sys.stdout.flush()
    if timed:
        tick = time()
    if verbose:
        print "Order ", omega
    print_stationary_spectrum_as_mma(alphabet, seq, omega, word_len, rec.description)
    if timed:
        print ""
        print "Elapsed: ", time() - tick
        print ""
    sys.stdout.flush()
    file_handle.close()
