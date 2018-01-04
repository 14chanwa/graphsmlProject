# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 11:06:07 2018

@author: Quentin
"""

import numpy as np
import scipy as sp
import networkx as nx
from graphSamplingWithDPP import generate_graph_from_stochastic_block_model
from graphSamplingWithDPP import generate_k_bandlimited_signal

"""
The goal of this test file is to demonstrate the computation difference between
different eigenvalue and eigenvector computation algorithms from Numpy and
Scipy.

"""


##### PARAMETERS #####

### Signal band
k = 3

### Graph creation parameters
N = 100              # Number of nodes
kgraph = 3          # Number of communities

c = 16               # Average degree

epsilonc = (c - np.sqrt(c)) / (c + np.sqrt(c) * (k - 1))    # Critical epsilon
                    # above which one can no longer distinguish communities
epsilon = 0.5 * epsilonc       # q2/q1


##### END PARAMETERS #####



# Generate graph
G = generate_graph_from_stochastic_block_model(N, kgraph, epsilon, c)

# Get laplacian and adjacency matrix
L = sp.sparse.csr_matrix(nx.laplacian_matrix(G), dtype='d')
W = sp.sparse.csr_matrix(nx.adjacency_matrix(G), dtype='d')

# Generate a k-bandlimited signal
x, alpha, Lambda_k, U_k = generate_k_bandlimited_signal(L, k)

# Using shift-inverse method
Lambda_k_ARPACK_shift_inverse, U_k_ARPACK_shift_inverse = sp.sparse.linalg.\
        eigsh(L, k=k, sigma=0, which='LM')

# Directly using SM from ARPACK (not efficient)
Lambda_k_ARPACK_SM, U_k_ARPACK_SM = sp.sparse.linalg.eigsh(L, k=k, which='SM')

# Reference: numpy function
Lambda_k_numpy_eigh, U_k_numpy_eigh = np.linalg.eigh(L.toarray())
idx = Lambda_k_numpy_eigh.argsort()   
Lambda_k_numpy_eigh = Lambda_k_numpy_eigh[idx[range(k)]]
U_k_numpy_eigh = U_k_numpy_eigh[:,idx[range(k)]]


print('----- Reference (Numpy) (not efficient):')
print('Lambda_k=', Lambda_k_numpy_eigh)
print('----- ARPACK shift-inverse:')
print('Lambda_k=', Lambda_k_ARPACK_shift_inverse)
print(np.dot(U_k_numpy_eigh.T, U_k_ARPACK_shift_inverse))
print('----- ARPACK SM (not efficient):')
print('Lambda_k=', Lambda_k_ARPACK_SM)
print(np.dot(U_k_numpy_eigh.T, U_k_ARPACK_SM))
print('----- Function chosen in generate_k_bandlimited_signal:')
print('Lambda_k=', Lambda_k)
print(np.dot(U_k_numpy_eigh.T, U_k))