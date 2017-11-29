# graphsmlProject

This repository implements some algorithms about Determinantal Point Processes for Graph Sampling, from the paper [Graph sampling with determinantal processes](https://arxiv.org/abs/1703.01594), N. Tremblay et al. (2017).


### Propp-Wilson algorithm and maze generation


The Propp-Wilson algorithm is an algorithm that enables sampling a random spanning tree of a directed graph (see [How to Get a Perfectly Random Sample from a Generic Markov Chain and Generate a Random Spanning Tree of a Directed Graph](https://www2.stat.duke.edu/~scs/Projects/Trees/Theory/ProppWilson1998.pdf), J.G. Propp and D.B. Wilson (1998) . It is possible to modify the initial algorithm to sample a Determinantal Point Process over the vertices of the graph, according to a certain kernel.


The algorithm is implemented in the file **wilson_algorithm.py**. A minimal working example is presented in the file **test_wilson_algorithm.py**.


A recreational application of the random spanning tree generation is the automated maze generation: let us interpret the walls of the maze as a covering tree branches, with the borders of the maze being the root of the tree ; then we are assured that the maze is feasible between any two points, and that there is one and only one path that do not involve stepping back that connects these two points. This application is implemented in file **generate_maze.py** and a minimal working example and plotting script is provided in **test_generate_maze.py**.

