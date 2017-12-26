# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 15:53:21 2017

@author: Quentin
"""

import numpy as np
import scipy as sp

def regularized_reweighted_recovery(L, pi, M, y, gamma, r):
    """
    Recover a k-bandlimited signal from m measurements on a subset of nodes
    sampled from a DPP.
    
    Parameters
    ----------
    L: sp.sparse.csr_matrix
        Laplacian of the graph.
    pi: np.array (N x 1)
        Array of marginal probabilities that i belongs to A (where A is the
        sampled measurement subset).
    M: sp.sparse.csr_matrix (m x N)
        Measurement matrix: each row is delta_{omega_j} where omega_j is the
        index of the sampled node corresponding to the column.
    y: np.array (m x 1)
        Array of the measurements of the signal.
    gamma: double
    r: int
    
    Returns
    ----------
    x_rec: np.array (N x 1)
        Reconstructed signal
    
    """
    
    # Direct inversion method
    Lr = L
    for i in range(r-1):
        Lr = Lr.dot(L)
    Pm1 = sp.sparse.diags(1/pi)

    tmp = ((M.transpose()).dot(Pm1).dot(M)).tocsr() + (gamma * Lr).tocsr()
    tmp2 = sp.sparse.linalg.inv(tmp.tocsc())
    
    xrec = tmp2.dot(M.transpose()).dot(Pm1).dot(y)
    
    # Gradient descent (TODO)
    
    return xrec
    