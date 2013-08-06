from random import randrange
from Bio import SeqIO

# Adapted from http://rosettacode.org/wiki/Knuth's_algorithm_S
# under GNU Free Documentation License 1.2
# http://www.gnu.org/licenses/fdl-1.2.html
class S_of_n_creator():
    def __init__(self, sample_size):
        self.sample_size = sample_size
        self.item_count = 0
        self.sample = []
 
    def __call__(self, item):
        self.item_count += 1
        sample_size, item_count, sample = self.sample_size, self.item_count, self.sample
        if item_count <= sample_size:
            # Keep first sample_size items
            sample.append(item)
        elif randrange(item_count) < sample_size:
            # Keep item
            sample[randrange(sample_size)] = item
        return sample

def sample_seqs_from_fasta(input_filename, output_filename, sample_size):
    s_of_n = S_of_n_creator(sample_size)
    input_handle = open(input_filename, "rU")
    output_handle = open(output_filename, "w")
    for record in SeqIO.parse(input_handle, "fasta"):
        sample = s_of_n(record)
    SeqIO.write(sample, output_handle, "fasta")
    input_handle.close()
    output_handle.close()
