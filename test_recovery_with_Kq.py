# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 18:06:48 2017

@author: Quentin
"""

from graphSamplingWithDPP import generate_graph_from_stochastic_block_model
from graphSamplingWithDPP import generate_k_bandlimited_signal
from graphSamplingWithDPP import wilson_algorithm
#from graphSamplingWithDPP import estimate_pi
from graphSamplingWithDPP import regularized_reweighted_recovery
from graphSamplingWithDPP import reweighted_recovery_with_eigendec

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
N = 100              # Number of nodes
kgraph = 3          # Number of communities

c = 14               # Average degree

epsilonc = (c - np.sqrt(c)) / (c + np.sqrt(c) * (k - 1))    # Critical epsilon
                    # above which one can no longer distinguish communities
epsilon = 0.5 * epsilonc       # q2/q1

### Number of measurements
m = 2 # should be >= k

### Initial q
q = 0.5             # Will be adapted if the size of the sampled DPP
                    # is not large enough.

# Recovery parameters
d = 20
n = 10 * int(np.floor(20 * np.log(N)))
gamma = 1e-5
r = 4

# Noise level
noise_sigma2 = 10e-4

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

# Estimate pi using fast graph filtering
#pi = estimate_pi(L, q, d, n)

## OR

# Theoretical pi using eigendecomposition
A = L.toarray()
V, U = np.linalg.eig(A)
g = q/(q+V)
gdiag = np.diag(g)
Kq = U.dot(gdiag).dot(U.transpose())
pi = np.diagonal(Kq)

pi_sample = pi[Y]


# Number of trial signals
nb_signals = 1000

# Results
results_known_Uk = list()

for i in range(nb_signals):

    # Generate a k-bandlimited signal
    x, alpha, Lambda_k, U_k = generate_k_bandlimited_signal(L, k)
    
    # Sample the signal
    M = sp.sparse.lil_matrix((len(Y), N))
    M[np.arange(len(Y)), Y] = 1
    M = M.tocsr()
    
    # Measurement + noise
    y = M.dot(x)
    y += np.random.normal(0, noise_sigma2, size=y.shape)
    
    ## Recovery with unknown U_k
    #xrec1 = regularized_reweighted_recovery(L, pi_sample, M, y, gamma, r)
    
    # Recovery with known U_k
    xrec2 = reweighted_recovery_with_eigendec(L, pi_sample, M, y, U_k)
    results_known_Uk.append(np.linalg.norm(x-xrec2))
    
print('--- Recovery with known Uk ---')
print('quantiles difference norm=', np.percentile(results_known_Uk, [10, 50, 90]))



## Recovery with unknown U_k
#xrec1 = regularized_reweighted_recovery(L, pi_sample, M, y, gamma, r)
#
## Recovery with known U_k
#xrec2 = reweighted_recovery_with_eigendec(L, pi_sample, M, y, U_k)
#
##print('x=', x)
#print('--- Recovery without Uk ---')
##print('xrec1=', xrec1)
#print('difference norm1=', np.linalg.norm(x-xrec1))
#print('--- Recovery with known Uk ---')
##print('xrec2=', xrec2)
#print('difference norm2=', np.linalg.norm(x-xrec2))