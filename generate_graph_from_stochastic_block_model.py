# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 17:07:39 2017

@author: Quentin
"""

import numpy as np
import networkx as nx

def generate_graph_from_stochastic_block_model(N, k, epsilon, c):
    
    """
    Generates a graph from the stochastic block model: given N nodes belonging
    to k communities, two nodes have a probability q1 of having a common edge 
    if they belong to the same community, or q2 if they do not.
    Specifying q1, q2 is equivalent to specifying epsilon = q2/q1 and the
    average degree c.
    See:
    Graph sampling with determinantal processes, Nicolas Tremblay et al., 2017
    
    Parameters
    ----------
    N: int
        Number of nodes.
    k: int
        Number of communities.
    epsilon: double
        Ratio q2/q1.
    c: double
        Average degree.
    
    Returns
    ----------
    G: nx.Graph
        The generated graph
    """
    
    q1 = c / ((N/k - 1) + epsilon * (N - N/k))
    q2 = epsilon * q1
    
    G = nx.Graph()
    G.add_nodes_from(range(0, N))
    
    # Community indices
    step = N/k
    indices = np.floor(np.arange(N)/step).astype(np.int)
    # print(indices)
    
    # Build edges
    for i in range(0, N):
        for j in range(i+1, N):
            if indices[i] == indices[j]:
                if np.random.rand() < q1:
                    G.add_edge(i, j)
            else:
                if np.random.rand() < q2:
                    G.add_edge(i, j)
    
    return G