# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 17:38:34 2017

@author: 14chanwa
"""

import numpy as np
import scipy as sp
import networkx as nx
from graphSamplingWithDPP import generate_graph_from_stochastic_block_model,\
    generate_k_bandlimited_signal, wilson_algorithm


"""
The goal of this test is to check whether wilson_algorithm returns a DPP with
the right parameters.
We generate a SBM graph and compute exactly K_q. We then check:
- Whether the empirical cardinal means to the expectation
- Whether the empirical probas of one singleton and one pair belonging to the
DPP converge to the theoretical probas
- Whether the norm of the reweighted measurement vector is equal in expectation
to the norm of the original signal.

"""

##### PARAMETERS #####

### Signal band
k = 2

### Graph creation parameters
N = 100              # Number of nodes
kgraph = 2          # Number of communities

c = 16               # Average degree

epsilonc = (c - np.sqrt(c)) / (c + np.sqrt(c) * (k - 1))    # Critical epsilon
                    # above which one can no longer distinguish communities
epsilon = 0.5 * epsilonc       # q2/q1


### DPP parameter
q = 1.0


##### END PARAMETERS #####



# Generate graph
G = generate_graph_from_stochastic_block_model(N, kgraph, epsilon, c)

# Get laplacian and adjacency matrix
L = sp.sparse.csr_matrix(nx.laplacian_matrix(G), dtype='d')
W = sp.sparse.csr_matrix(nx.adjacency_matrix(G), dtype='d')


# Generate a k-bandlimited signal
x, alpha, Lambda_k, U_k = generate_k_bandlimited_signal(L, k)


# Build K_q
A = L.toarray()
V, U = np.linalg.eigh(A)
g = q/(q+V)
gdiag = np.diag(g)
K_q = U.dot(gdiag).dot(U.transpose())

# Check DPP sample validity
# For some singleton
singleton = 80
singleton2 = 12
# and some pair
pair = np.array([80, 81])
pair2 = np.array([40, 70])

# Check prop III.1 (E(norm(P^{-1/2} * M * x)^2) = norm(x)^2)
sq_norms = []

# Check cardinal
card = []

# Number of iterations
n_iterations = 10000
singleton_count= 0
singleton2_count= 0
pair_count = 0
pair2_count = 0

print('n_iterations=', n_iterations)

for i in range(n_iterations):
    # Sample from DPP
    Ycal = wilson_algorithm(W, q)[0]
    #    print('Ycal=', Ycal)
    #    print(np.in1d(singleton, np.array(Ycal)))
    #    print(np.in1d(pair, np.array(Ycal)))
    
    if np.in1d(singleton, np.array(Ycal)).all():
        singleton_count += 1
    if np.in1d(singleton2, np.array(Ycal)).all():
        singleton2_count += 1
    if np.in1d(pair, np.array(Ycal)).all():
        pair_count += 1
    if np.in1d(pair2, np.array(Ycal)).all():
        pair2_count += 1
        
    Pm12 = np.diag(1 / np.sqrt(np.diagonal(K_q)[Ycal]))
    M = np.zeros((len(Ycal), N))
    M[np.arange(len(Ycal)), Ycal] = 1
    sq_norms.append(np.linalg.norm(Pm12.dot(M).dot(x))**2)
    card.append(len(Ycal))
    
print('------- Cardinal')
print('Theoretical=', np.sum(g))
print('Empirical=', np.mean(card))
print('------- Singleton:')
print('Theoretical proba=', K_q[singleton, singleton])
print('Empirical proba=', singleton_count / n_iterations)
print('------- Singleton2:')
print('Theoretical proba=', K_q[singleton2, singleton2])
print('Empirical proba=', singleton2_count / n_iterations)
print('------- Pair:')
print('Theoretical proba=', np.linalg.det((K_q[:, pair])[pair, :]))
print('Empirical proba=', pair_count / n_iterations)
print('------- Pair2:')
print('Theoretical proba=', np.linalg.det((K_q[:, pair2])[pair2, :]))
print('Empirical proba=', pair2_count / n_iterations)
print('------- Norms:')
print('mean np.linalg.norm(x)**2=', np.linalg.norm(x)**2)
print('mean np.linalg.norm(Pm12.dot(M).dot(x))**2=', np.mean(sq_norms))

