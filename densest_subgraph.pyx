# cython: language_level=3
# cython: linetrace=True
# distutils: language=c++

import os.path as path
import numpy as np
import matplotlib.pyplot as plt
import sys

from time import time
from sklearn.metrics import r2_score

cimport cython
cimport numpy as np
#from libcpp.vector cimport vector


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
    cdef double number_of_edges = 0

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

                u = int(edges[0])
                v = int(edges[1])

                line = file.readline()

                if u == v:
                    continue

                adj_lists[u].append(v)
                adj_lists[v].append(u)
                deg[u] += 1
                deg[v] += 1
                number_of_edges += 1

            return number_of_edges, deg, adj_lists

    else:
        raise Exception("Invalid specified path : " + file_path)


# O(n+m)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef densest_linear_test(int n,
                          double number_of_edges,
                          int[:] deg,
                          list adj_lists):
    """
    Computes the densest subgraph. 
    :param n: the number of nodes.
    :param number_of_edges: the computed number of edges in the graph.
    :param deg: a list with the degree of each node.
    :param adj_lists: filled adjacency lists.
    :return: The densest subgraph as a list, its density and number of edges as int.
    """
    # O(1)
    if n <= 0 :
        raise Exception("The number of nodes can't be less than 1")

    # O(n)
    cdef list deg_list = [[] for _ in range(n)]
    cdef list removed_nodes = []

    cdef int[:] chosen_nodes = np.ones(n, dtype=int)
    cdef int[:] node_used = np.zeros(n, dtype=int)

    cdef int compteur = 0
    cdef int optimal_state = 0
    cdef int min_deg, temp_deg, node, erased_edges, neighbour, found

    cdef double temp_roh, roh_h

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


def test_temps(wanted = "temps"):
    """
    Simple function returning the time used to compute on several graphs.
    :param: wanted : string to know if I want to plot times or density.
    :return:
    """
    # n for roadNet-PA is not the one given on internet but an estimate of the maximum label of a node in the file.
    n = np.array([4039, 5908, 7057, 13866, 14113, 27917, 50515, 3892, 41773, 54573, 47538, 1091000])
    m = np.array([88234, 41729, 89455, 86858, 52310, 206259, 819306, 17262, 125826, 498202, 222887, 1541898])
    labels = np.array(["ego-Facebook", "gemsec_politician", "gemsec_government", "gemsec_athletes", "gemsec_company",
                       "gemsec_new_sites", "gemsec_artists", "gemsec_tvshows", "gemsec_RO", "gemsec_HR", "gemsec_HU",
                       "roadNet-PA"])
    rho = np.zeros(len(n))

    #n = np.array([4039, 7057, 27917, 50515, 1091000])
    #m = np.array([88234, 89455, 206259, 819306, 1541898])
    #labels = np.array(["ego-Facebook", "gemsec_government", "gemsec_new_sites", "gemsec_artists", "roadNet-PA"])

    densest_times = np.zeros(len(n))
    whole_times = np.zeros(len(n))

    for i in range(len(n)):
        whole_start = time()
        number_of_edges, deg, adj_lists = filling_adjacency(labels[i] + ".txt", n[i])
        start_time = time()
        _,rho[i],_ = densest_linear_test(n[i], number_of_edges, deg, adj_lists)
        densest_times[i] = time() - start_time
        whole_times[i] = time() - whole_start

    if wanted == "density":
        return n, m, rho

    return n + m, densest_times, whole_times

def plotting_times():
    """
    Function plotting the times used to compute on several graphs.
    :return:
    """
    n_m, densest_times, whole_times = test_temps()

    coef_dens = np.polyfit(n_m, densest_times,1)
    coef_whole = np.polyfit(n_m, whole_times,1)

    # attemps to fit a degree one polynome y = a * x + b.
    poly1d_dens = np.poly1d(coef_dens)
    poly1d_whole = np.poly1d(coef_whole)
    # poly1d_fn is now a function which takes in x and returns an estimate for y
    prediction_dens = poly1d_dens(n_m)
    prediction_whole = poly1d_whole(n_m)

    r2_dens = r2_score(densest_times, prediction_dens)
    r2_whole = r2_score(whole_times, prediction_whole)

    # Plotting the figure
    plt.clf()
    plt.plot(n_m, densest_times, "bs", label="Computing densest subgraph")
    plt.plot(n_m, prediction_dens, "b", label="R²= "+ str(r2_dens))
    plt.plot(n_m, whole_times, "r^", label="Computing + filling adjacency lists")
    plt.plot(n_m, prediction_whole, "r", label= "R²= " + str(r2_whole))
    plt.xlabel("Sum of n and m")
    plt.ylabel("Time in seconds")
    plt.title("Time to compute the densest subgraph approximation on several graphs.")
    plt.legend()
    plt.show()

def plotting_density():
    """
    Function plotting the densities.
    :return:
    """
    n, m, rho = test_temps("density")
    x = np.arange(len(n))

    # Plotting the figure
    plt.clf()
    plt.plot(x, m/n, "bs", label="Starting density")
    plt.plot(x, rho, "r^", label="Found density")
    plt.xlabel("Different graphs")
    plt.ylabel("Density")
    plt.title("Initial density and best found density on several graphs.")
    plt.legend()
    plt.show()