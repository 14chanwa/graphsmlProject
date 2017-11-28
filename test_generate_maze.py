# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 18:01:02 2017

@author: 14chanwa
"""

# Generates a maze from the Propp-Wilson algorithm.

import numpy as np
from generate_maze import generate_maze

n = 25
m = 10

P, W = generate_maze(n, m)

# TODO: implement trees in order not to have a list of lists as walls (not very
# elegant...)

# Print maze
# For each branch, print branch

import matplotlib.pyplot as plt 

# Borders
plt.plot([-1, -1], [-1, m], color = 'black')
plt.plot([-1, n], [-1, -1], color = 'black')
plt.plot([-1, n], [m, m], color = 'black')
plt.plot([n, n], [-1, m], color = 'black')

# Plot walls
# We decide arbitrary orientation of the walls in the corners
# where there is an ambiguity
for ell in P:
    for i in range(np.size(ell)-1):
        y1 = ell[i] % m
        x1 = ell[i] // m
        y2 = ell[i+1] % m
        x2 = ell[i+1] // m
        if ell[i+1] == n*m:
            if x1 == 0:
                x2 = -1
                y2 = y1
            elif x1 == n-1:
                x2 = n
                y2 = y1
            elif y1 == 0:
                y2 = -1
                x2 = x1
            elif y1 == m-1:
                y2 = m
                x2 = x1
        plt.plot([x1, x2], [y1, y2], color = 'black')

## Nodes
## In practice, it seems it is easier to solve the maze if these are plotted
#nodes = np.zeros([n*m, 2])
#for i in range(0, n):
#    nodes[i*m:((i+1)*m), 0] = i * np.ones(m)
#    nodes[i*m:((i+1)*m), 1] = np.arange(m)
#plt.scatter(nodes[:, 0], nodes[:, 1])

plt.gca().set_aspect('equal', adjustable='box')
plt.show()

#Y, P = wilson_algorithm(W, 0.1)
