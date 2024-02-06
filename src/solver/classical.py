from itertools import combinations

import networkx as nx


def find_max_clique(G: nx.Graph) -> list:
    """Finds the maximum clique with brute force."""
    cliques = list(nx.find_cliques(G))
    largest_clique = max(cliques, key=len)
    return largest_clique


def get_mis(graph) -> list[list]:
    """Finds the MIS with brute force."""
    independent_sets = _get_all_independent_sets(graph)
    mis_size = len(max(independent_sets, key=len))
    mis = [
        list(independent_set)
        for independent_set in independent_sets
        if len(independent_set) == mis_size
    ]
    return mis


def _get_all_independent_sets(graph):
    all_nodes = set(graph.nodes())
    independent_sets = []

    for r in range(1, len(all_nodes) + 1):
        for subset in combinations(all_nodes, r):
            if all(graph.has_edge(u, v) is False for u, v in combinations(subset, 2)):
                independent_sets.append(set(subset))

    return independent_sets
