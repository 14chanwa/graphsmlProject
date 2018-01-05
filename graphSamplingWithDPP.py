# -*- coding: utf-8 -*-
"""
Created on Sat Dec 30 18:17:27 2017

@author: Quentin
"""

import numpy as np
import scipy as sp
import networkx as nx


###############################################################################
# Graph generation
###############################################################################


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
    #print('q1=', q1)
    #print('q2=', q2)
    
    G = nx.Graph()
    G.add_nodes_from(range(0, N))
    
    # Community indices
    #step = N/k
    #indices = np.floor(np.arange(N)/step).astype(np.int)
    #print(indices)
    div, mod = divmod(N, k)
    indices = np.repeat(np.arange(k), [div+1 if i<mod else div for i in \
                        range(k)])
    
    # Build edges
    L = []      # edges list
    for i in range(0, N):
        for j in range(i+1, N):
            if indices[i] == indices[j]:
                if np.random.rand() < q1:
                    L.append((i, j))
            else:
                if np.random.rand() < q2:
                    L.append((i, j))
    
    G.add_edges_from(L)
    
    return G


###############################################################################
# Signal sampling
###############################################################################


def generate_k_bandlimited_signal(L, k, Lambda_k=None, U_k=None):
    
    """
    Generates a k-bandlimited signal (a signal that is the combination of the
    first k eigenmodes of the Laplacian, i.e. the k modes with the smallest
    eigenvalues).
    
    Parameters
    ----------
    L: sp.sparse.csr_matrix (N x N)
        Laplacian matrix of the graph.
    k: int
        The sampled signal will be a combination of the k first eigenmodes.
    (Optional) Lambda_k: np.array (k x 1)
        The k-lowest eigenvalues of L. If not provided, will compute.
    (Optional) U_k: np.array (N x k)
        The corresponding k eigenmodes. If not provided, will compute.
    
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
    
    if Lambda_k is None or U_k is None:
        # Find the k lowest eigenmodes using Scipy shift-invert mode
        # See: https://docs.scipy.org/doc/scipy/reference/tutorial/arpack.html
        Lambda_k, U_k = sp.sparse.linalg.eigsh(L, k=k, sigma=0, which='LM')
    alpha = np.random.normal(0, 1, (k,))
    alpha /= np.linalg.norm(alpha)
    x = U_k.dot(alpha)
    
    return x, alpha, Lambda_k, U_k


###############################################################################
# DPP sampling
###############################################################################


def sample_from_DPP(Lambda, V):
    
    """
    Sample from a DPP from the kernel K of eigendecomposition (Lambda, V).
    This is Algorithm 2 in 
    Graph sampling with determinantal processes, Nicolas Tremblay et al., 2017
    and Algorithm 1 in
    Determinantal Point Processes for Machine Learning, A. Kulesza et al., 2012
    
    Parameters
    ----------
    Lambda: np.array (N x 1)
        Eigenvalues of K.
    V: np.array (N x N)
        Eigenvectors of K.
    
    Returns
    ----------
    Ycal: list
        List of indices of the sampled nodes
    """
    
    N = Lambda.size
    
    #    Lambda = np.real(Lambda)
    
    # Sample some indices
    J = np.random.rand(len(Lambda)) < Lambda
    
    # Select eigenvectors from indices
    #    Vcal = np.real(V[:, J])
    Vcal = V[:, J]
    Ycal = list()
    
    while Vcal.shape[1] > 0:
        # Choose an index with probability 1/card(V) * sum_v (v' * e_i)^2
        P = np.sum(np.square(Vcal), axis=1)#\
            #.reshape(-1)
        P = P / Vcal.shape[1]
        index = np.random.choice(N, p=P)
        Ycal.append(index)
#        
#        if Vcal.shape[1] > 1:     
#            # Make V an orthonormal basis of V orthogonal to e_index using 
#            # Gram-Schmidt algorithm
#            Vnew = np.zeros(Vcal.shape)
#            # First base vector: sampled node
#            Vnew[index, 0] = 1
#            for j in range(1, Vcal.shape[1]):
#                u = Vcal[:, j]
#                for k in range(j):
#                    u -= Vcal[:, j].dot(Vnew[:, k]) * Vnew[:, k]
#                Vnew[:, j] = u / np.linalg.norm(u)
#            # Only keep the orthogonal to the first vector
#            Vcal = Vnew[:, 1:Vcal.shape[1]]
#        else:
#            break

        if Vcal.shape[1] > 1:
            # Find Vcal[:, j] that is a linear combination of e_index
            j = np.where(Vcal[index,:]!=0)[0][0]
            # Remove some Vcal[:, j] to cancel the contribution of V along 
            # e_index (outputs some base of V orthogonal to e_index)
            Vcal -= (Vcal[:,j]/Vcal[index,j])[:,np.newaxis] * Vcal[index,:] 
            # Delete the column j
            Vcal = np.delete(Vcal, j, axis=1)
            # Normalize the first column
            Vcal[:, 0] /= np.linalg.norm(Vcal[:, 0])
            # Use Gram-Schmidt to orthonormalize the rest of the vectors
            for j in range(1, Vcal.shape[1]):
                Vcal[:, j] -= np.sum(Vcal[:, j].transpose().dot(Vcal[:, 1:j]) \
                    * Vcal[:, 1:j], axis=1)
                Vcal[:, j] /= np.linalg.norm(Vcal[:, j])
        else:
            break
            
    return Ycal


def estimate_pi(L, q, d, n):
    
    """
    Estimate pi_i, the probabilities that nodes i belong to the set of sampled
    nodes, without computing the eigendecomposition of L, using fast filtering
    on graphs, under the DPP of kernel Kq.
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
    lambda_N = sp.sparse.linalg.eigsh(L, k=1, return_eigenvectors=False, \
                                      which='LA')[0]
    
    
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


def wilson_algorithm(W, q=0):
    
    """
    Implements Propp-Wilson algorithm to sample a random spanning tree from the
    directed graph of adjacency matrix W, and a DPP with marginal kernel
    Kq = U * g_q (Lambda) * U' where g_q (lambda) = lambda / (lambda + q)
    as described in:
    Graph sampling with determinantal processes, Nicolas Tremblay et al., 2017
    
    Parameters
    ----------
    W : scipy.sparse.csr_matrix
        Adjacency matrix of the graph. The diagonal elements do not matter.
        The csr_matrix format is adapted for this task as we need fast row
        slicing.
    q : float
        Weight of the sink node (Delta).
    
    Returns
    ----------
    Y : set
        Set of node indices that correspond to the sampled DPP. This is the set
        of the nodes such that the random walk process led to the sink at the
        following step.
    P : list(list(int))
        A list of lists of node indices that depict the paths taken during the 
        random walk process. The union of the nodes constitute a random
        spanning tree.
    
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
            transition_probabilities = np.asarray(W.getrow(walk_index)\
                                                  .todense()).reshape(-1)
            
            # Get sum of weights and normalize
            # Add the sink probability to position walk_index
            transition_probabilities[walk_index] = 0
            normalization_weight = np.sum(transition_probabilities) + q

            transition_probabilities[walk_index] = q
            
            # If the probability transition is null (case q = 0 and leaf of the
            # graph), finish
            if normalization_weight == 0 or normalization_weight == 0:
                #print('case 0')
                for j in walk:
                    Nu[j] = 1
                P.append(walk)
                break
            
            transition_probabilities /= normalization_weight
            
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


def generate_maze(n, m=None):
    
    """
    Generates a maze of wall sizes n * m with the Propp-Wilson algorithm.
    
    Parameters
    ----------
    n: int
        Number of wall rows of the maze. Should be > 1.
    m: int (defaults to n)
        Number of wall columns of the maze. Should be > 1.
    
    Returns
    ----------
    P: list(list(int))
        A list of paths such that the union of the paths is a randomly sampled
        undirected spanning tree for the maze wall graph, that is the set of 
        walls of the maze. The frontier of the maze is represented by node n*m,
        the other nodes are ordered by lines.
    W: scipy.sparse.csr_matrix
        The adjacency matrix used to generate the maze.
    
    """
    
    if m == None:
        m = n
    
    # Create a graph with n * m + 1 nodes
    # The last node is the frontier of the maze
    # Diagonal elements are sum of probabilities (1)
    
    # Construct W as a dok_matrix
    W = sp.sparse.eye(n*m+1, format="dok")
    W[n*m, n*m] = 0
    
    # Corners of the maze
    # [row, column]
    # [0, 0]
    W[0, 1] = 1/3
    W[0, m] = 1/3
    W[0, n*m] = 1/3
    # [n-1, 0]
    W[(n-1)*m, (n-1)*m + 1] = 1/3
    W[(n-1)*m, (n-2)*m] = 1/3
    W[(n-1)*m, n*m] = 1/3
    # [0, m-1]
    W[m-1, m-2] = 1/3
    W[m-1, 2*m-1] = 1/3
    W[m-1, n*m] = 1/3
    # [n-1, m-1]
    W[n*m-1, n*m-2] = 1/3
    W[n*m-1, (n-1)*m-1] = 1/3
    W[n*m-1, n*m] = 1/3
    
    # Upper frontier
    for j in range(1,m-1):
        W[j, j-1] = 1/4
        W[j, j+1] = 1/4
        W[j, m + j] = 1/4
        W[j, n*m] = 1/4
    # Bottom frontier
    for j in range(1,m-1):
        W[(n-1)*m+j, (n-1)*m+j-1] = 1/4
        W[(n-1)*m+j, (n-1)*m+j+1] = 1/4
        W[(n-1)*m+j, (n-2)*m+j] = 1/4
        W[(n-1)*m+j, n*m] = 1/4
    # Left frontier
    for i in range(1,n-1):
        W[i*m, (i-1)*m] = 1/4
        W[i*m, (i+1)*m] = 1/4
        W[i*m, i*m+1] = 1/4
        W[i*m, n*m] = 1/4
    # Right frontier
    for i in range(1,n-1):
        W[(i+1)*m-1, i*m-1] = 1/4
        W[(i+1)*m-1, (i+2)*m-1] = 1/4
        W[(i+1)*m-1, (i+1)*m-2] = 1/4
        W[(i+1)*m-1, n*m] = 1/4
    
    # Other nodes
    for i in range(1, n-1):
        for j in range(1, m-1):
            W[i*m+j, i*m+j-1] = 1/4
            W[i*m+j, i*m+j+1] = 1/4
            W[i*m+j, (i-1)*m+j] = 1/4
            W[i*m+j, (i+1)*m+j] = 1/4
    
    # Convert W to a csr_matrix
    W = W.tocsr()
    
    # Run Propp-Wilson algorithm on W
    Y, P = wilson_algorithm(W)
    
    return P, W


###############################################################################
# Signal reconstruction
###############################################################################


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

    # For consistency, Uk should be in Scipy format
    Uk = sp.sparse.csr_matrix(Uk)
    
    # Let the signal be written as x = Uk * alpha
    # Then the perfect reconstruction formula writes:
    # alpha = (Uk' M' P^-1 M Uk)^-1 Uk' M' P^-1 y
    alpha = sp.sparse.linalg.inv(Uk.transpose().dot(M.transpose()).dot(Pm1)\
                .dot(M).dot(Uk)).dot(Uk.transpose().dot(M.transpose())\
                    .dot(Pm1).dot(y))
    
    # OR
    
    # Gradient descent (to do)
    
    return Uk.dot(alpha)


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
    
    Pm1 = sp.sparse.diags(1/pi)
    
    # Direct inversion method using the formula
    # x_rec = (M' P^-1 M + gamma L^r)^-1 M' P^-1 y 
    Lr = L
    for i in range(r-1):
        Lr = Lr.dot(L)

    tmp = ((M.transpose()).dot(Pm1).dot(M)).tocsc() + (gamma * Lr).tocsc()
    tmp2 = sp.sparse.linalg.inv(tmp)
    
    xrec = tmp2.dot(M.transpose()).dot(Pm1).dot(y)
    
    ## OR
    
    # Gradient descent (TODO)
    
    return xrec


