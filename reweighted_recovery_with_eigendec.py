# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 19:44:30 2017

@author: Quentin
"""

import scipy as sp


def reweighted_recovery_with_eigendec(L, pi, M, y, Uk):
    
    """
    Recover a k-bandlimited signal from m measurements on a subset of nodes
    sampled from a DPP, knowing the spanning eigenspace.
    
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
    Uk: sp.sparse.csr_matrix (N, m)
        Eigenvectors spanning the signal.
    
    Returns
    ----------
    x_rec: np.array (N x 1)
        Reconstructed signal
    
    """
    
    Pm1 = sp.sparse.diags(1/pi)

    Uk = sp.sparse.csr_matrix(Uk)
    alpha = sp.sparse.linalg.inv(Uk.transpose().dot(M.transpose()).dot(Pm1).dot(M).dot(Uk)).dot(Uk.transpose().dot(M.transpose()).dot(Pm1).dot(y))
    return Uk.dot(alpha)