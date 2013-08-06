#!/usr/bin/python
import bioarray as ba
import numpy as np
mat = np.ones((4,4)) * 0.25
ba.print_array_as_mmc(mat, "DNA, iid")
