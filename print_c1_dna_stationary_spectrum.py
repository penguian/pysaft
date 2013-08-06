#!/usr/bin/python
import bioarray as ba
import sys
omega     = (int(sys.argv[1]) 
                 if len(sys.argv) > 1 else
             3
            )  
word_len  = (int(sys.argv[2]) 
                 if len(sys.argv) > 2 else 
             5
            ) 
file_name = (sys.argv[3] 
                 if len(sys.argv) > 3 else 
             "/home/leopardi/Download/ensembl/Homo_sapiens.GRCh37.69.dna_sm.chromosome.1.fa"
            )
ba.print_dna_stationary_spectrum_as_mma(file_name, omega, word_len, 
                                        masked=True, verbose=False, timed=False)
