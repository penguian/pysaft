#!/usr/bin/python
import sample_seqs as ss
import sys
input_file_name  = sys.argv[1]
sample_size      = (int(sys.argv[2]) 
                        if len(sys.argv) > 2 else 
                    100
                   ) 
if len(sys.argv) > 3:
    output_file_name = sys.argv[3]
else:
    prefix, dot, suffix = input_file_name.rpartition('.')
    output_file_name = prefix + '.' + str(sample_size) + '.' + suffix 
ss.sample_seqs_from_fasta(input_file_name, output_file_name, sample_size)