# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 18:06:48 2017

@author: Quentin
"""

from generate_graph_from_stochastic_block_model import generate_graph_from_stochastic_block_model
from generate_k_bandlimited_signal import generate_k_bandlimited_signal
from wilson_algorithm import wilson_algorithm
from estimate_pi import estimate_pi
from regularized_reweighted_recovery import regularized_reweighted_recovery

import numpy as np
import scipy as sp
import networkx as nx


# A complete usecase
# Create a graph and a k-bandlimited signal on this graph
# Measure the signal on a sample subset of nodes
# Try to reconstruct the signal


##### PARAMETERS #####

### Signal band
k = 2

### Graph creation parameters
N = 10              # Number of nodes
kgraph = 2          # Number of communities

c = 8               # Average degree

epsilonc = (c - np.sqrt(c)) / (c + np.sqrt(c) * (k - 1))
epsilon = 1 * epsilonc       # q2/q1

### Number of measurements
m = 5 # should be >= k

### Initial q
q = 1.0             # Will be adapted if the size of the sampled DPP
                    # is not large enough.

# Recovery parameters
d = 15
n = int(np.floor(20 * np.log(N)))
gamma = 1e-5
r = 4

##### END PARAMETERS #####



# Generate graph
G = generate_graph_from_stochastic_block_model(N, kgraph, epsilon, c)

# Get laplacian and adjacency matrix
L = sp.sparse.csr_matrix(nx.laplacian_matrix(G), dtype='d')
W = sp.sparse.csr_matrix(nx.adjacency_matrix(G), dtype='d')

# Generate a k-bandlimited signal
x, alpha, Lambda_k, U_k = generate_k_bandlimited_signal(L, k)

# Sample m nodes from a DPP
# Adapt q using a heuristic from
# On some random forests with determinantal roots, L. Avena, A. Gaudillère
Y = []
while len(Y) < m:
    if len(Y) > 0:
        q = q * m / len(Y)
    Y = wilson_algorithm(W, q)[0]
    print('Attempting to sample m nodes... q=', q)
    print('Size of the sample=', len(Y))
Y.sort()
print('Sampled DPP=', Y)

# Sample the signal
M = sp.sparse.lil_matrix((len(Y), N))
M[np.arange(len(Y)), Y] = 1
M = M.tocsr()

# Measurement + noise
y = M.dot(x)
y += np.random.normal(0, 10e-4, size=y.shape)

# Recover the signal
pi = estimate_pi(L, q, d, n)
pi_sample = pi[Y]
xrec = regularized_reweighted_recovery(L, pi_sample, M, y, gamma, r)

print('x=', x)
print('xrec=', xrec)
print('difference norm=', np.linalg.norm(x-xrec))
