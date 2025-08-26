from itertools import combinations
from typing import Any

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


def greedy_max_weight_clique(
    graph: nx.Graph,
    weight_attr: str = "weight",
    min_clique_size: int = 2,
) -> list[Any]:
    """
    Approximate a maximum weight clique in a node-weighted undirected graph using a greedy heuristic.

    The algorithm starts from the highest-weight node and incrementally builds cliques
    by greedily adding compatible neighbors with the highest weights.

    Args:
        graph (nx.Graph): Undirected NetworkX graph with node weights.
        weight_attr (str): Node attribute name containing weights.
        min_clique_size (int): Minimum size for a valid clique.

    Returns:
        list[Any]: List of nodes forming the approximate maximum weight clique.
    """
    best_clique: list[Any] = []
    best_weight: float = 0.0

    # Sort nodes by weight descending
    sorted_nodes = sorted(
        graph.nodes,
        key=lambda n: graph.nodes[n].get(weight_attr, 0),
        reverse=True,
    )

    for start in sorted_nodes:
        clique = [start]
        candidate_nodes = set(graph.neighbors(start))

        while candidate_nodes:
            # Filter candidates that are connected to all current clique members
            valid = [n for n in candidate_nodes if all(graph.has_edge(n, c) for c in clique)]
            if not valid:
                break

            # Choose the best next node by weight
            next_node = max(valid, key=lambda n: graph.nodes[n].get(weight_attr, 0))
            clique.append(next_node)
            candidate_nodes.intersection_update(graph.neighbors(next_node))

        if len(clique) >= min_clique_size:
            weight = sum(graph.nodes[n].get(weight_attr, 0) for n in clique)
            if weight > best_weight:
                best_weight = weight
                best_clique = clique

    return best_clique


import networkx as nx


def find_exhaustive_weighted_max_clique(graph: nx.Graph) -> tuple[list[str] | None, float]:
    """
    Finds the maximum weight clique in a networkx graph using an exhaustive
    backtracking search (Bron-Kerbosch with pivoting and weighting).

    It assumes that each node has a 'weight' attribute.

    Args:
        graph: A networkx.Graph object where each node has a 'weight' attribute.

    Returns:
        A tuple containing:
        - The list of nodes in the highest-weight clique found.
        - The total weight of that clique.
    """

    # --- State variables for the search, stored in a mutable dict ---
    solution_tracker = {"best_clique": [], "max_weight": 0.0}

    # --- The core recursive backtracking function ---
    def find_cliques_recursive(
        current_clique: list[str],
        candidates: set[str],
        excluded: set[str],
    ):
        """
        Recursively finds maximal cliques and updates the best solution found.
        """
        # Pruning Step: Stop if this path cannot possibly beat the best score.
        current_weight = sum(graph.nodes[n]["weight"] for n in current_clique)
        potential_from_candidates = sum(graph.nodes[n]["weight"] for n in candidates)
        if current_weight + potential_from_candidates <= solution_tracker["max_weight"]:
            return

        # Base case: If no more candidates, we have a maximal clique.
        if not candidates and not excluded:
            if current_weight > solution_tracker["max_weight"]:
                solution_tracker["max_weight"] = current_weight
                solution_tracker["best_clique"] = current_clique.copy()
            return

        # Pivot selection for performance improvement.
        try:
            pivot = next(iter(candidates | excluded))
        except StopIteration:
            return  # No more nodes to check

        # Iterate through candidates that are NOT neighbors of the pivot.
        candidates_to_process = list(candidates - set(graph.neighbors(pivot)))

        for node in candidates_to_process:
            neighbors_of_node = set(graph.neighbors(node))

            # Recursive call with the new extended clique
            find_cliques_recursive(
                current_clique + [node],
                candidates & neighbors_of_node,
                excluded & neighbors_of_node,
            )

            # Backtrack: Move the processed node from candidates to excluded.
            candidates.remove(node)
            excluded.add(node)

    # --- Initial Call to start the search ---
    all_graph_nodes = set(graph.nodes())
    find_cliques_recursive([], all_graph_nodes, set())

    best_found_clique = solution_tracker["best_clique"]
    if not best_found_clique:
        return None, 0.0

    return sorted(best_found_clique), solution_tracker["max_weight"]
