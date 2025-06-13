from __future__ import annotations
from typing import Any, Dict, Hashable, Set, Tuple
import networkx as nx


class WeightedMIS:
    """
    Solver for the exact weighted Maximum Independent Set (MIS) problem on undirected graphs.
    Uses a basic branch-and-bound algorithm with weight-based bounding.
    """

    def __init__(self, graph: nx.Graph, weight: str = "weight") -> None:
        """
        Initializes the solver.

        Args:
            graph (nx.Graph): An undirected NetworkX graph where each node has a weight attribute.
            weight (str): The node attribute key for weights. Defaults to "weight".
        """
        self._graph: nx.Graph = graph
        self._weight_attr: str = weight
        self._best_set: Set[Hashable] = set()
        self._best_weight: float = 0.0

    def solve(self) -> Tuple[Set[Hashable], float]:
        """
        Computes the exact weighted MIS.

        Returns:
            Tuple[Set[Hashable], float]: A tuple containing the set of selected nodes and the total weight.
        """
        nodes = list(self._graph.nodes)
        # Precompute weights
        weights: Dict[Hashable, float] = {
            n: float(self._graph.nodes[n].get(self._weight_attr, 1.0)) for n in nodes
        }
        # Sort nodes for branch ordering (largest weight first)
        nodes.sort(key=lambda n: weights[n], reverse=True)
        self._branch_and_bound(set(), set(nodes), weights)
        return self._best_set, self._best_weight

    def _branch_and_bound(
        self, current_set: Set[Hashable], candidates: Set[Hashable], weights: Dict[Hashable, float]
    ) -> None:
        """
        Recursive branch-and-bound search.

        Args:
            current_set (Set[Hashable]): Currently selected independent set.
            candidates (Set[Hashable]): Remaining candidate nodes to consider.
            weights (Dict[Hashable, float]): Precomputed node weights.
        """
        # Compute an upper bound: sum of all candidate weights
        bound = sum(weights[n] for n in candidates)
        if sum(weights[n] for n in current_set) + bound <= self._best_weight:
            return

        if not candidates:
            current_weight = sum(weights[n] for n in current_set)
            if current_weight > self._best_weight:
                self._best_weight = current_weight
                self._best_set = set(current_set)
            return

        # Select next node to branch on
        node = max(candidates, key=lambda n: weights[n])

        # Branch: include node
        new_set = set(current_set)
        new_set.add(node)
        # Exclude neighbors of node
        new_candidates = candidates - {node} - set(self._graph.neighbors(node))
        self._branch_and_bound(new_set, new_candidates, weights)

        # Branch: exclude node
        candidates.discard(node)
        self._branch_and_bound(current_set, candidates, weights)


def solve_weighted_mis(graph: nx.Graph, weight: str = "weight") -> Tuple[Set[Any], float]:
    """
    Convenience function to compute the exact weighted Maximum Independent Set.

    Args:
        graph (nx.Graph): An undirected NetworkX graph with node weights.
        weight (str): Node attribute name for weights. Defaults to "weight".

    Returns:
        Tuple[Set[Any], float]: The optimal set of nodes and its total weight.
    """
    solver = WeightedMIS(graph, weight)
    return solver.solve()
