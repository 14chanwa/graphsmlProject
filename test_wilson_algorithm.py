# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 17:38:34 2017

@author: 14chanwa
"""

# Tests Propp-Wilson algorithm on a simple matrix.

import numpy as np
from scipy.sparse import csr_matrix
from graphSamplingWithDPP import wilson_algorithm


# Original algorithm
# Testing wilson_algorithm on a simple 3x3 matrix. We expect only one path of
# length 3, and Y = emptyset (no vertex is sampled).
W = 0.5 * np.ones([3, 3])
for j in range(0, 3):
    W[j, j] = 1

W = csr_matrix(W)
print(W)

Y, P = wilson_algorithm(W)
print(Y)
print(P)


# Modified algorithm
# If we add a significant weight on the sink, there can be several paths of
# length < 3 (and thus some vertices are sampled).

Y, P = wilson_algorithm(W, 0.5)
print(Y)
print(P)

