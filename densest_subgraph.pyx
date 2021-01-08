# cython: language_level=3
# cython: linetrace=True
# distutils: language=c++

import os.path as path
import numpy as np
import matplotlib.pyplot as plt
import sys
import cProfile

from time import time
from sklearn.metrics import r2_score

cimport cython
cimport numpy as np
from libcpp.vector cimport vector


if "win" in sys.platform:
    DATA_PATH = ".\\data"
else:
    DATA_PATH = "./data"


# O(m)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef filling_adjacency(str filename,
                       int n):
    """
    Method to fill the adjacency with the file containing edges.
    :param filename: Given filename as a string.
    :param n: Number of nodes.
    :return: Initialised degree and adjaceny lists or raises an issue if the filename is incorrect.
    """
    cdef int[:] deg = np.zeros(n, dtype=int)
    cdef list adj_lists = [[] for _ in range(n)]
    cdef str file_path = ""
    cdef str line = ""
    cdef int u, v
    cdef list edges

    # O(1), more accurate to say something that will be tiny in front of n or m.
    file_path = path.join(DATA_PATH, filename)

    # O(m)
    if path.exists(file_path) and path.isfile(file_path):
        with open(file_path, 'r') as file:
            line = file.readline()

            # O(m)
            while not line =="":
                # All this in O(1)
                edges = line.strip().split(" ")

                # TODO don't use a cast here or check
                u = int(edges[0])
                v = int(edges[1])

                line = file.readline()

                if u == v:
                    continue

                adj_lists[u].append(v)
                adj_lists[v].append(u)
                deg[u] += 1
                deg[v] += 1

            return deg, adj_lists

    else:
        raise Exception("Invalid specified path : " + file_path)


# O(n+m)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef densest_linear_test(int n,
                          int m,
                          str filename):
    """
    Computes the densest subgraph. 
    :param n: the number of nodes.
    :param m: number of edges.
    :param filename : the name of the file
    :return: The densest subgraph as a list, its density and number of edges as int.
    """
    # O(1)
    if n <= 0 :
        raise Exception("The number of nodes can't be less than 1")

    # O(n)
    cdef list deg_list = [[] for _ in range(n)]
    cdef list removed_nodes = []
    cdef list adj_lists

    cdef int[:] deg
    cdef int[:] chosen_nodes = np.ones(n, dtype=int)
    cdef int[:] node_used = np.zeros(n, dtype=int)

    cdef int compteur = 0
    cdef int optimal_state = 0
    cdef int roh_h, min_deg, temp_deg, node, erased_edges, neighbour, found
    cdef int number_of_edges = m

    cdef double temp_roh

    # O(m)
    deg, adj_lists = filling_adjacency(filename, n)

    # O(1)
    # Setting initial rho_h
    rho_h = number_of_edges / n

    # O(n)
    # Setting initial degrees state.
    min_deg = n - 1
    for i in range(n):
        temp_deg = deg[i]
        deg_list[temp_deg].append(i)
        min_deg = min(min_deg, temp_deg)

    # O(m) pour moi
    # We erase at most n - 1 nodes.
    while compteur < n - 1:
        node = -1
        found = 0
        # O(??) the issue is here. I think it's ok since we add
        # each node at most deg of it in the list. I think summed on
        # the whole n - 1 loops it must be O(m).
        while found == 0:
            # O(1)
            if deg_list[min_deg]:
                node = (deg_list[min_deg]).pop()
                # Checking if the node has already been used.
                found = 1 - node_used[node]
            # O(1)
            else:
                # Otherwise just go further in the chain.
                min_deg += 1

        # Shouldn't happen yet better safe than sorry.
        if node == - 1:
            raise Exception("Issue when trying to find next node to delete, none found.")

        # Found a node to compute :
        node_used[node] = True
        compteur += 1
        removed_nodes.append(node)

        # O(deg(node))
        erased_edges = 0

        for i in range(len(adj_lists[node])):
            neighbour = adj_lists[node][i]
            if not node_used[neighbour]:
                # O(1)
                erased_edges += 1
                deg[neighbour] -= 1
                min_deg = min(min_deg, deg[neighbour])
                deg_list[deg[neighbour]].append(neighbour)

        # Check if we have better density
        # O(1)
        number_of_edges -= erased_edges
        temp_roh = number_of_edges / (n - compteur)
        if temp_roh > rho_h:
            rho_h = temp_roh
            optimal_state = compteur
            optimal_edges = number_of_edges

    # Need to compute H
    # O(n)

    for i in range(optimal_state):
        chosen_nodes[removed_nodes[i]] = False

    # O(n)
    return [i for i in range(n) if chosen_nodes[i]], rho_h, optimal_edges


def test_temps():
    #n = np.array([4039, 7057, 13866, 27917, 50515, 1090950])
    #m = np.array([88234, 89455, 86858, 206259, 819306, 1541898])
    #labels = np.array(["ego-Facebook", "gemsec_government", "gemsec_athletes", "gemsec_new_sites", "gemsec_artists",
                       #"roadNet-PA"])

    n = np.array([4039, 5908, 7057, 13866, 14113, 27917, 50515, 3892, 41773, 54573, 47538])
    m = np.array([88234, 41729, 89455, 86858, 52310, 206259, 819306, 17262, 125826, 498202, 222887])
    labels = np.array(["ego-Facebook", "gemsec_politician", "gemsec_government", "gemsec_athletes", "gemsec_company",
                       "gemsec_new_sites", "gemsec_artists", "gemsec_tvshows", "gemsec_RO", "gemsec_HR", "gemsec_HU"])

    times = np.zeros(len(n))
    start_time = time()

    for i in range(len(n)):
        _,_,_ = densest_linear_test(n[i], m[i], labels[i] + ".txt")
        times[i] = time()

    for i in range(len(times) - 1, 0, -1):
        times[i] -= times[i - 1]
    times[0] -= start_time
    return n + m, times

def plotting_times():
    n_m, times = test_temps()

    coef = np.polyfit(n_m, times,1)
    # attemps to fit a degree one polynome y = a * x + b.
    poly1d_fn = np.poly1d(coef)
    # poly1d_fn is now a function which takes in x and returns an estimate for y
    prediction = poly1d_fn(n_m)

    r2 = r2_score(times, prediction)
    print(r2)
    # Plotting the figure
    plt.clf()
    plt.plot(n_m, times, "+r")
    plt.plot(n_m, prediction, "b")
    plt.show()