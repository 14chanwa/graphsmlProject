# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 12:52:19 2017

@author: Quentin
"""

import numpy as np
import scipy as sp

def estimate_pi(L, q, d, n):
    
    """
    Estimate pi_i, the probabilities that nodes i belong to the set of sampled
    nodes, without computing the eigendecomposition of L, using fast filtering
    on graphs.
    This is Algorithm 4 in 
    Graph sampling with determinantal processes, Nicolas Tremblay et al., 2017
    
    Parameters
    ----------
    L: scipy.sparse.csr_matrix
        Laplacian of the graph
    q: double
        Weigh of the sink of the graph
    d: int
        Degree of the polynomial approximation of sqrt(g_q)
    n: int
    
    Returns
    ----------
    pi: np.array
        Vector of pi's
    """
    
    # Size of the laplacian
    N = L.shape[0]
    
    # Compute the largest eigenvalue of L
    lambda_N = sp.sparse.linalg.eigsh(L, k=1, return_eigenvectors=False, which='LM')[0]
    
    # Polynomial approximation of sqrt(g_q)
    x = np.arange(0, lambda_N + lambda_N/d, lambda_N/d)
    w = np.sqrt(q/(q + x))
    p = sp.interpolate.lagrange(x, w)
    beta = p.c[::-1]
    
    # Generate normal random variables
    # R is of shape (N, n)
    R = np.random.normal(0, 1/n, (N, n))
    
    # Recursively compute SR
    LT = L.transpose()
    Lpow = L
    # SR is of shape (N, n)
    SR = beta[0] * sp.sparse.eye(N) + beta[1] * Lpow
    for ell in range(2, d+1):
        Lpow = Lpow.dot(LT)
        SR = SR + beta[ell] * Lpow
    SR = SR.dot(R)
    # print(SR.shape)
        
    # Compute pi
    # Elevate to square
    SR **= 2
    # Sum over rows
    pi = SR.sum(axis = 1) 
    
    return pi #, beta
    
    