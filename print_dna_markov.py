import markov as m
file_name = "/home/leopardi/Downloads/ensembl/Homo_sapiens.GRCh37.69.dna_sm.chromosome.1.fa"
m.print_dna_markov_arrays(file_name, 5, masked=True)
