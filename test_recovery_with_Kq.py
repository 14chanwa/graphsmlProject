# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 18:06:48 2017

@author: Quentin
"""

import numpy as np
import scipy as sp
import networkx as nx

from graphSamplingWithDPP import generate_graph_from_stochastic_block_model,\
    generate_k_bandlimited_signal, wilson_algorithm,\
    getmatrix_reweighted_recovery_with_eigendec


# A complete usecase
# Create a graph and a k-bandlimited signal on this graph
# Measure the signal on a sample subset of nodes
# Try to reconstruct the signal


##### PARAMETERS #####

### Signal band
k = 2

### Graph creation parameters
N = 100              # Number of nodes
kgraph = 2          # Number of communities

c = 16               # Average degree

epsilonc = (c - np.sqrt(c)) / (c + np.sqrt(c) * (kgraph - 1))    # Critical 
               # epsilon above which one can no longer distinguish communities
epsilon = 0.1 * epsilonc       # q2/q1

### Number of measurements
m = 2 # should be >= k

### Initial q
initial_q = 0.2             # Will be adapted if the size of the sampled DPP
                    # is not large enough.

# Noise levels
noise_sigma = np.array([1e-6, 1e-5, 1e-4, 1e-3, 1e-2])

##### END PARAMETERS #####


# Number of different graphs
nb_graphs = 100
print('Nb graph trials=', nb_graphs)

# Number of trial signals per graph
nb_signals = 100
print('Nb signal trials per graph=', nb_signals)


# Results
results_known_Uk = np.empty((len(noise_sigma), 0)).tolist()
sample_cardinal = list()

for j in range(nb_graphs):

    # Generate graph
    G = generate_graph_from_stochastic_block_model(N, kgraph, epsilon, c)
    # Check that the graph is completely connected
    while nx.number_connected_components(G) > 1:
        G = generate_graph_from_stochastic_block_model(N, kgraph, epsilon, c)
    
    # Get laplacian and adjacency matrix
    L = sp.sparse.csr_matrix(nx.laplacian_matrix(G), dtype='d')
    W = sp.sparse.csr_matrix(nx.adjacency_matrix(G), dtype='d')
    
    # Compute U_k
    x, alpha, Lambda_k, U_k = generate_k_bandlimited_signal(L, k)
    
    # Sample m nodes from a DPP
    # Adapt q using a heuristic from
    # On some random forests with determinantal roots, L. Avena, A. Gaudillère
    Y = []
    q = initial_q
    while len(Y) < k or m - 2 * np.sqrt(m) > len(Y) \
             or m + 2 * np.sqrt(m) < len(Y):
        if len(Y) > 0:
            q = q * m / len(Y)
        Y = wilson_algorithm(W, q)[0]
        print('Attempting to sample m nodes... q=', q)
        print('Size of the sample=', len(Y))
    Y.sort()
    print('Sampled DPP=', Y)
    sample_cardinal.append(len(Y))
    
    # Theoretical pi using eigendecomposition
    A = L.toarray()
    V, U = np.linalg.eigh(A)
    g = q/(q+V)
    gdiag = np.diag(g)
    Kq = U.dot(gdiag).dot(U.transpose())
    pi = np.diagonal(Kq)
    pi_sample = pi[Y]
    
    # Sampling matrix
    M = sp.sparse.lil_matrix((len(Y), N))
    M[np.arange(len(Y)), Y] = 1
    M = M.tocsr()
    
    # Get reconstruction matrix (only depends on the graph and the sample)
    T = getmatrix_reweighted_recovery_with_eigendec(L, pi_sample, M, U_k)
    
    for i in range(nb_signals):
    
        # Generate a k-bandlimited signal
        # Use optional arguments in order not to recompute the eigendec
        x, alpha, Lambda_k, U_k = generate_k_bandlimited_signal(L, k, \
                                            Lambda_k = Lambda_k, U_k = U_k)
        
        # Measurement + noise
        y = M.dot(x)
        for noise_index in range(len(noise_sigma)):
            nl = noise_sigma[noise_index]
            y += np.random.normal(0, nl, size=y.shape)
            
            
            # Recovery with known U_k
            xrec = T.dot(y)
            
            if np.linalg.norm(x-xrec) > 1:
                print('--- anormaly detected, normdiff=', np.linalg.norm(x-xrec))
                print('Y=', Y)
            
            results_known_Uk[noise_index].append(np.linalg.norm(x-xrec))

#%%
error_means = np.zeros(len(noise_sigma))
error_bars = np.zeros((2, len(noise_sigma)))

print('mean cardinal=', np.mean(sample_cardinal))

for noise_index in range(len(noise_sigma)):
    nl = noise_sigma[noise_index]
    print('--- Recovery with known Uk ---')
    print('noise_level=', nl)
    print('10, 50, 90 quantiles difference norm=')
    percentiles = np.percentile(results_known_Uk[noise_index], [50, 10, 90])
    print(percentiles)
    print('max difference norm=', np.max(results_known_Uk[noise_index]))

    error_means[noise_index] = percentiles[0]
    error_bars[0, noise_index] = percentiles[0] - percentiles[1]
    error_bars[1, noise_index] = percentiles[2] - percentiles[0]

import matplotlib.pyplot as plt 
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.errorbar(noise_sigma, error_means, yerr=error_bars)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_ylabel('$\|x - x_{rec}\|_2$')
ax.set_xlabel('Noise $\sigma$')
ax.set_title('Error in function of the noise using $K_q$ ($\epsilon=0.1*\epsilon_c$)')
#plt.savefig("project_report\error_function_noise_Kq.eps", format="eps")
plt.show()
