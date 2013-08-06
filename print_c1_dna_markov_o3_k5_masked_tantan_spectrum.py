#!/usr/bin/python
import bioarray as ba
import sys
file_name = "/home/leopardi/Download/ensembl/Homo_sapiens.GRCh37.69.dna_sm.chromosome.1.fa"
if len(sys.argv) > 1:
    file_name = sys.argv[1]
omega = 3
word_len = 5 
ba.print_dna_stationary_spectrum(file_name, omega, word_len, masked=True, verbose=False, timed=False)
