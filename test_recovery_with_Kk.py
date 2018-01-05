# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 18:06:48 2017

@author: Quentin
"""

from graphSamplingWithDPP import generate_graph_from_stochastic_block_model,\
    generate_k_bandlimited_signal, sample_from_DPP,\
    reweighted_recovery_with_eigendec,\
    regularized_reweighted_recovery

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
kgraph = 2        # Number of communities

c = 16               # Average degree

epsilonc = (c - np.sqrt(c)) / (c + np.sqrt(c) * (k - 1))    # Critical epsilon
                    # above which one can no longer distinguish communities
epsilon = 0.1 * epsilonc       # q2/q1

# Recovery parameters
d = 30
n = int(np.floor(20 * np.log(N)))
gamma = 1e-5
r = 4

# Noise level
noise_sigma = 10e-5

##### END PARAMETERS #####

# Number of different graphs
nb_graphs = 100
print('Nb graph trials=', nb_graphs)

# Number of trial signals per graph
nb_signals = 100
print('Nb signal trials per graph=', nb_signals)

# Results
results_known_Uk = list()

for j in range(nb_graphs):

    # Generate graph
    G = generate_graph_from_stochastic_block_model(N, kgraph, epsilon, c)
    #    print('Number of CCs=', nx.number_connected_components(G))
    
    # Get laplacian and adjacency matrix
    L = sp.sparse.csr_matrix(nx.laplacian_matrix(G), dtype='d')
    W = sp.sparse.csr_matrix(nx.adjacency_matrix(G), dtype='d')
    
    # Compute U_k
    x, alpha, Lambda_k, U_k = generate_k_bandlimited_signal(L, k)
    
    # Sample nodes from the DPP of kernel K_k
    K_k = U_k.dot(U_k.transpose())
    eigenvalues, eigenvectors = np.linalg.eigh(K_k)
    Y = sample_from_DPP(eigenvalues, eigenvectors)
#    Y = dpp_exact_sampling_KuTa12(eigenvalues, eigenvectors)
    #    print('Sampled DPP=', Y)
    
    # Theoretical pi using eigendecomposition
    pi = np.diagonal(K_k)
    
    pi_sample = pi[Y]
    
    for i in range(nb_signals):
    
        # Generate a k-bandlimited signal
        x, alpha, Lambda_k, U_k = generate_k_bandlimited_signal(L, k)
        
        # Sample the signal
        M = sp.sparse.lil_matrix((len(Y), N))
        M[np.arange(len(Y)), Y] = 1
        M = M.tocsr()
        
        # Measurement + noise
        y = M.dot(x)
        y += np.random.normal(0, noise_sigma, size=y.shape)
        
        # Recovery with unknown U_k
#        xrec2 = regularized_reweighted_recovery(L, pi_sample, M, y, gamma, r)
        
#        # Recovery with known U_k
        xrec2 = reweighted_recovery_with_eigendec(L, pi_sample, M, y, U_k)
        
        if np.linalg.norm(x-xrec2) > 1:
            print('--- anormaly detected, normdiff=', np.linalg.norm(x-xrec2))
            print('Y=', Y)
#            print('x=', x)
#            print('xrec=', xrec2)
        
        results_known_Uk.append(np.linalg.norm(x-xrec2))
    
print('--- Recovery with known Uk ---')
print('10, 50, 90 quantiles difference norm=')
print(np.percentile(results_known_Uk, [10, 50, 90]))
print('max difference norm=', np.max(results_known_Uk))
print('expected noise norm=', np.linalg.norm(np.random.normal(0, noise_sigma,\
                                                              size=k)))