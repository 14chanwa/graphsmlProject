# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 23:34:45 2018

@author: Quentin
"""


import numpy as np
import networkx as nx
from graphSamplingWithDPP import generate_graph_from_stochastic_block_model

"""
Checks whether the function generate_graph_from_stochastic_block_model
generates suitable graphs: checks whether q1, q2, c are respected.
"""


### Graph creation parameters
N = 50000            # Number of nodes
kgraph = 2           # Number of communities

c = 16               # Average degree

epsilonc = (c - np.sqrt(c)) / (c + np.sqrt(c) * (kgraph - 1))    # Critical epsilon
                    # above which one can no longer distinguish communities
epsilon = 0.5 * epsilonc       # q2/q1

G = generate_graph_from_stochastic_block_model(N, kgraph, epsilon, c)
A = nx.adjacency_matrix(G)
#print(A)

q1 = c / ((N/kgraph - 1) + epsilon * (N - N/kgraph))
q2 = epsilon * q1

div, mod = divmod(N, kgraph)
indices = np.repeat(np.arange(kgraph), [div+1 if i<mod else div for i in \
                        range(kgraph)])

# Check whether q1, q2, c are respected for a given node
test_community = 0
q1_count = 0
q2_count = 0
degrees = list()
for test_node in np.where(indices == test_community)[0]:
    test_row = A[test_node, :].toarray().reshape(-1)
    q1_count += np.sum(test_row[indices == indices[test_node]])
    q2_count += np.sum(test_row[indices != indices[test_node]])
    degrees.append(np.sum(test_row))
    
print('N=', N)
print('empirical q1=', q1_count / (len(test_row[indices == \
        indices[test_node]]) * len(test_row[indices != indices[test_node]])))
print('theoretical q1=', q1)

print('empirical q2=', q2_count / (len(test_row[indices == \
        indices[test_node]]) * len(test_row[indices != indices[test_node]])))
print('theoretical q2=', q2)

print('empirical degree=', np.mean(degrees))
print('theoretical degree=', c)