# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 17:38:34 2017

@author: 14chanwa
"""

# Tests Propp-Wilson algorithm on a simple matrix.

exec(open("./wilson_algorithm.py").read())

# Testing wilson_algorithm
W = 0.5 * np.ones([3, 3])
for j in range(0, 3):
    W[j, j] = 1

print(W)

Y, P = wilson_algorithm(W)

#print(W)
print(Y)
print(P)