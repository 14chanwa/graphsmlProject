# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 14:17:21 2017

@author: Quentin
"""

import numpy as np
import scipy as sp

from graphSamplingWithDPP import estimate_pi
import networkx as nx
from graphSamplingWithDPP import generate_graph_from_stochastic_block_model

#N = 100
#q = 30
#d = 5
#
## Generate a random graph
#G = nx.fast_gnp_random_graph(N, 0.6)
#
## Get the laplacian of the graph (converted to double)
#L = sp.sparse.csr_matrix(nx.normalized_laplacian_matrix(G), dtype='d')
#
## Compute an expected value of card(A)
#ev = sp.sparse.linalg.eigsh(L, k = int(N/4), return_eigenvectors=False)
#print(ev)
#print(np.mean(q/(q+ev))*N)
#
## Estimate the marginal probabilities that node i is in the set of nodes 
## sampled by the DPP of kernel K_q
#p = estimate_pi(L, q, d, int(np.floor(20 * np.log(N))))
#
#print(p)

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


A = L.toarray()
V, U = np.linalg.eig(A)
idx = V.argsort()
V = V[idx]
U = U[idx]

# Compute DPP kernel K_q
g = q/(q+V)
G = np.diag(g)
Kq = U.dot(G).dot(U.transpose())

pi_th = np.diagonal(Kq)
pi_exp = estimate_pi(L, q, 10, int(20 * np.log(N)))

print('pi_th=', pi_th)
print('pi_exp=', pi_exp)
