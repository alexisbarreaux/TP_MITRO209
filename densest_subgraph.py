import os.path as path
import sys

# Dataset notes :
# ego-Facebook : n = 4039, m = 88234, "ego-Facebook.txt"
# H = densest_linear_test(4039, "ego-Facebook.txt")
# gemsec_artists : n = 50515, m= 819306, "gemsec_artists.txt"
# gemsec_government : n = 7057, m = 89455, "gemsec_government.txt"
# H = densest_linear_test(7057, "gemsec_government.txt")
# to-graph : n = 5, edges = 

if "win" in sys.platform:
    DATA_PATH = ".\\data"
else:
    DATA_PATH = "./data"


# O(m)
def filling_adjacency(filename, adj_lists=None, deg=None, number_of_edges=0):
    """
    Method to fill the adjacency with the file containing edges.
    :param filename: Given filename as a string.
    :param adj_lists: Adjacency list.
    :param deg
    :param number_of_edges
    :return: The number of edges if it went well, -1 otherwise.
    """
    # O(1)
    if deg is None:
        deg = []
    if adj_lists is None:
        adj_lists = []

    # O(1), more accurate to say something that will be tiny in front of n or m.
    file_path = path.join(DATA_PATH, filename)

    # O(m)
    if path.exists(file_path) and path.isfile(file_path):
        with open(file_path, 'r') as file:
            line = file.readline()

            # O(m)
            while not len(line) == 0:
                # All this in O(1)
                line = line.replace("  ", " ")
                edges = line.strip().split(" ")

                # TODO don't use a cast here or check
                u, v = int(edges[0]), int(edges[1])
                adj_lists[u].append(v)
                adj_lists[v].append(u)
                deg[u] += 1
                deg[v] += 1
                number_of_edges += 1

                line = file.readline()

            return number_of_edges

    else:
        print("Invalid specified path : " + file_path)
        return -1


# O(n+m)
def densest_linear_test(n, filename):
    """
    Computes the densest subgraph.
    :param n: the number of nodes.
    :param filename: A list of list of edges.
    :return: the densest subgraph.
    """

    # O(n)
    adj_lists = [[] for _ in range(n)]
    deg = [0 for _ in range(n)]
    deg_list = [[] for _ in range(n)]
    node_used = [False for _ in range(n)]
    compteur = 0
    number_of_edges = 0
    removed_nodes = []
    optimal_state = 0

    # O(m)
    number_of_edges = filling_adjacency(filename, adj_lists, deg, number_of_edges)
    print("n edges : ", number_of_edges)
    if number_of_edges < 0:
        # TODO raise smth here
        return []

    # O(1)
    # Setting initial ro_h
    ro_h = number_of_edges / n
    print(ro_h)

    # O(n)
    # Setting initial degrees state.
    min_deg = n - 1
    for i in range(n):
        deg_list[deg[i]].append(i)
        min_deg = min(min_deg, deg[i])
    print(min_deg)
    print("deg", deg_list[min_deg])
    # O(m) pour moi
    # We erase at most n - 1 nodes.
    while compteur < n - 1:
        found = False
        node = -1
        # TODO I must check this, it is the sole true issue for me.
        # O(??) the issue is here. I think it's ok since we add
        # each node at most deg of it in the list. I think summed on
        # the whole n - 1 loops it must be O(m).
        while not found:
            # O(1)
            if len(deg_list[min_deg]) != 0:
                node = (deg_list[min_deg]).pop()
                found = not node_used[node]
            # O(1)
            else:
                # no more edges. Normally can't happen, since
                # we will compute each node and read compteur == n- 1
                # before having this issue
                if min_deg == n - 1:
                    found = True
                # Otherwise just go further in the chain.
                else:
                    min_deg += 1

        # Shouldn't happen yet better safe than sorry.
        if node == - 1:
            return []

        # Found a node to compute :
        compteur += 1
        removed_nodes.append(node)

        # O(deg(node))
        # TODO j'ai l'impression qu'il y a un souci avec le nombre d'arcs et leur maj
        erased_edges = 0
        for neighbour in adj_lists[node]:
            if not node_used[neighbour]:
                # O(1)
                erased_edges += 1
                deg[neighbour] -= 1
                min_deg = min(min_deg, deg[neighbour])
                deg_list[deg[neighbour]].append(neighbour)

        # Maj possible de ro_h
        # O(1)
        number_of_edges -= erased_edges
        print(number_of_edges)
        a = number_of_edges / (n - compteur)
        if a > ro_h:
            ro_h = a
            optimal_state = compteur

    # Need to compute H
    # O(n)

    chosen_nodes = [True for _ in range(n)]
    for i in range(optimal_state):
        chosen_nodes[removed_nodes[i]] = False

    # O(n)
    print(ro_h, optimal_state)
    return [i for i in range(n) if chosen_nodes[i]]

