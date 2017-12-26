# graphsmlProject

This repository implements some algorithms about Determinantal Point Processes for Graph Sampling, from the paper [Graph sampling with determinantal processes](https://arxiv.org/abs/1703.01594), N. Tremblay et al. (2017).


### Propp-Wilson algorithm and maze generation


The Propp-Wilson algorithm is an algorithm that enables sampling a random spanning tree of a directed graph (see [How to Get a Perfectly Random Sample from a Generic Markov Chain and Generate a Random Spanning Tree of a Directed Graph](https://www2.stat.duke.edu/~scs/Projects/Trees/Theory/ProppWilson1998.pdf), J.G. Propp and D.B. Wilson (1998)). It is possible to modify the initial algorithm to sample a Determinantal Point Process over the vertices of the graph, according to a certain kernel.


The algorithm is implemented in the file **wilson_algorithm.py**. A minimal working example is presented in the file **test_wilson_algorithm.py**.


A recreational application of the random spanning tree generation is the automated maze generation: let us interpret the walls of the maze as a covering tree branches, with the borders of the maze being the root of the tree ; then we are assured that the maze is feasible between any two points, and that there is one and only one path that do not involve stepping back that connects these two points. This application is implemented in file **generate_maze.py** and a minimal working example and a plotting script are provided in **test_generate_maze.py**.


### Graph generation from the Stochastic Block Model


The stochastic block model is a community-structured graph model. Given `N` nodes belonging to `k` different communities, two nodes have a probability `q1` of sharing an edge if they belong to the same community and `q2` else. One shows that it is equivalent to define `epsilon=q2/q1` and `c` the average degree. The generation of such graphs is implemented in **generate_graph_from_stochastic_block_model.py**.


### k-bandlimited signal reconstruction with direct formula


This pipeline is implemented in **main_test_file.py**. We generate a graph using the Stochastic Block Model, then generate a `k`-bandlimited signal on this graph using the first `k` eigenmodes. We then sample a DPP of the correct size using the Propp-Wilson algorithm and use it to measure the signal on the graph. Finally, we use a direct formula (which implies the inversion of a matrix) to compute the reconstructed signal.
