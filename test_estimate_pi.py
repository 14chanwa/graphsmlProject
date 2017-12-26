# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 14:17:21 2017

@author: Quentin
"""

import numpy as np
import scipy as sp

from estimate_pi import estimate_pi
import networkx as nx

N = 100
q = 30
d = 10

# Generate a random graph
G = nx.fast_gnp_random_graph(N, 0.6)

# Get the laplacian of the graph (converted to double)
L = sp.sparse.csr_matrix(nx.normalized_laplacian_matrix(G), dtype='d')

# Compute an expected value of card(A)
ev = sp.sparse.linalg.eigsh(L, k = int(N/4), return_eigenvectors=False)
print(ev)
print(np.mean(q/(q+ev))*N)

# Estimate the marginal probabilities that node i is in the set of nodes 
# sampled by the DPP of kernel K_q
p = estimate_pi(L, q, d, int(np.floor(20 * np.log(N))))

print(p)
