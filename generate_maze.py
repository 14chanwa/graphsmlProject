# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 17:54:54 2017

@author: 14chanwa
"""

exec(open("./wilson_algorithm.py").read())


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
    W: np.array
        The adjacency matrix used to generate the maze.
    
    """
    
    if m == None:
        m = n
    
    # Create a graph with n * m + 1 nodes
    # The last node is the frontier of the maze
    # Diagonal elements are sum of probabilities (1)
    W = np.eye(n*m+1)
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
    
    # Run Propp-Wilson algorithm on W
    Y, P = wilson_algorithm(W)
    
    return P, W