# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 17:50:17 2017

@author: Quentin
"""

import numpy as np
import scipy as sp

def generate_k_bandlimited_signal(L, k):
    
    """
    Generates a k-bandlimited signal.
    
    Parameters
    ----------
    L: sp.sparse.csr_matrix (N x N)
        Laplacian matrix of the graph.
    k: int
        The sampled signal will be a combination of the k first eigenmodes.
    
    Returns
    ----------
    x: np.array (N x 1)
        The sampled signal.
    alpha: np.array (k x 1)
        The coefficients of x in the base of the k first eigenmodes.
    Lambda_k: np.array (k x 1)
        The k-lowest eigenvalues of L.
    U_k: np.array (N x k)
        The corresponding k eigenmodes.
    """
    
    Lambda_k, U_k = sp.sparse.linalg.eigsh(L, k=k)
    alpha = np.random.normal(0, 1, (k,))
    alpha /= np.linalg.norm(alpha)
    x = U_k.dot(alpha)
    
    return x, alpha, Lambda_k, U_k