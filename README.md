# A Comparison of Parallel Graph Coloring Algorithms
## Introduction
In general, graph coloring can refer to conditionally labelling any component of a graph such as its vertices or edges. We deal with a special case of graph coloring called "<b>Vertex Coloring</b>". The problem statement is as follows:

An undirected graph G is a set of vertices V and a set of edges E. The edges are of the form ```(i, j)``` where i,j belong to V . A coloring of a graph G is a mapping c : ```V -> {1, 2,..., s}``` such that ```c(i) != c(j)``` for all edges ```(i, j)``` belonging to E. c(i) is referred to as the color of vertex i.

Finding the <b>optimal</b> graph coloring is an <i>NP-Complete</i> problem. The goal of this project is to obtain a balanced coloring of the graph i.e, the number of colors used is made as close as possible to the <b>chromatic number</b> (minimum number for that graph) to ensure some degree of load balancing.

## Graph representation


## Approach
The serial implementation for the algorithm is an iterative greedy algorithm that is strictly sequential making parallelisation considerably difficult. However by assigning weights to each vertex based on certain constraints, parallelisation can be made possible. In the <b>Maximal Independent Set</b> approach, colors the graph by repeatedly finding the largest possible independent set of vertices in the graph. All vertices in the first such set are given the same color and removed from the graph. The algorithm then finds a new MIS and gives these a second color, and continues finding and coloring maximal independent sets until all vertices have been colored. In the <b>Jones Plassman</b> approach, vertices are initially given random weights and each vertex is colored with the smallest available color if it has a greater weight than all its neighbors. In the <b>Largest Degree First </b>algorithm, the vertex is colored if it has the highest degree in the set containing itself and its neighbors. In the <b>Smallest Degree First</b> algorithm, the initial weighting process is done on the basis of degree. In each algorithm, the crucial portion is to resolve conflicts, that is to remove all the instances of wrong coloring (neighbors sharing the same color).

## Dependencies


## Achieved Speedup : 
