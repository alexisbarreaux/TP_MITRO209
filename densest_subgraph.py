# O(n+m)
def densest_linear_test(n, file):
    """
    Computes the densest subgraph.
    :param n: the number of nodes.
    :param file: A list of list of edges.
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
    for [u, v] in file:
        adj_lists[u].append(v)
        adj_lists[v].append(u)
        deg[u] += 1
        deg[v] += 1
        number_of_edges += 1

    # O(1)
    # Setting initial ro_h
    ro_h = number_of_edges / n

    # O(n)
    # Setting initial degrees state.
    min_deg = n - 1
    for i in range(n):
        deg_list[deg[i]].append(i)
        min_deg = min(min_deg, deg[i])

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
                else:
                    min_deg += 1

        # Shouldn't happen yet better safe than sorry.
        if node == - 1:
            return []

        # Found a node to compute :
        compteur += 1
        removed_nodes.append(node)

        # O(deg(node))
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
    return [i for i in range(n) if chosen_nodes[i]]
