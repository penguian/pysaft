#!/usr/bin/python
import bioarray as ba
import numpy as np
word_len = 5
omega = 1
mat = np.ones((4,4)) * 0.25
alpha = len(ba.dna_alphabet)
spec = ba.build_stationary_spectrum(4, omega, mat, word_len)
header_line = "DNA, iid, word length ", str(word_len)
ba.print_array_as_mma(spec, header_line)
