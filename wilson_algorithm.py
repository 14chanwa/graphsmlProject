# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 23:37:54 2017

@author: 14chanwa
"""

import numpy as np


def wilson_algorithm(W, q=0):
    
    """
    Implements Propp-Wilson algorithm to sample a random spanning tree from the
    directed graph of adjacency matrix W, and a DPP with marginal kernel
    Kq = U * g_q (Lambda) * U' where g_q (lambda) = lambda / (lambda + q)
    as described in:
    Graph sampling with determinantal processes, Nicolas Tremblay et al., 2017
    
    Parameters
    ----------
    W : np.array
        Adjacency matrix of the graph. Assume W[i, i] = sum of outgoing edges.
    q : float
        Weight of the sink node (Delta)
    
    Returns
    ----------
    Y : set
        Set of node indices that correspond to the sampled DPP. This is the set
        of the nodes such that the random walk process led to the sink at the
        following step.
    P : set(List)
        A set of Lists that depict the paths taken during the random 
        walk process.
    
    """
    
    # Number of nodes
    n = W.shape[0]
    
    Y = list()
    P = list()
    
    Nu = np.zeros(n, dtype=bool)
    
    while np.sum(Nu) < n:
        
        # Select unvisited index
        walk_index = 0
        for j in range(0, n):
            if not Nu[j]:
                walk_index = j
                #Nu[j] = 1
                break
        
        # Start a random walk starting from node index
        # Keep indices in a list and in a boolean vector (for commodity)
        walk = list()
        walk_indices = np.zeros(n, dtype=bool)
        walk.append(walk_index)
        walk_indices[walk_index] = 1
        
        while True:
            # Get transition weights
            transition_probabilities = np.copy(W[walk_index, :])
            
            # Get sum of weights and normalize
            # Add the sink probability to position walk_index
            normalization_weight = transition_probabilities[walk_index] + q

            transition_probabilities[walk_index] = q
            transition_probabilities /= normalization_weight
            
            # If the probability transition is null (case q = 0 and leaf of the
            # graph), finish
            if normalization_weight == 0 or sum(transition_probabilities) == 0:
                #print('case 0')
                for j in walk:
                    Nu[j] = 1
                P.append(walk)
                break
            
            # Get random walk transition
            next_index = np.random.choice(n, p=transition_probabilities)

            # If we end up in the sink, add node to Y, add path to Nu and quit
            if next_index == walk_index:
                #print('case 1')
                Y.append(walk_index)
                for j in walk:
                    Nu[j] = 1
                P.append(walk)
                break
            
            # If we end in a node in Nu, add path to Nu and quit
            elif Nu[next_index]:
                #print('case 2')
                for j in walk:
                    Nu[j] = 1
                walk.append(next_index)
                P.append(walk)
                break
            
            # If we loop over ourselves, erase the entire loop
            elif walk_indices[next_index]:
                #print('case 3')
                # Handle the case when this is the last node (happens when the
                # entire graph is crossed in only one walk)
                if sum(walk_indices) + sum(Nu) == n:
                    #print('case 3a')
                    for j in walk:
                        Nu[j] = 1
                    P.append(walk)
                    break
                #print('case 3b')
                current_index = walk.pop()
                while current_index != next_index:
                    walk_indices[current_index] = 0
                    current_index = walk.pop()
            
            # Else continue walk
            walk_index = next_index
            walk.append(next_index)
            walk_indices[next_index] = 1
            
    return Y, P
    