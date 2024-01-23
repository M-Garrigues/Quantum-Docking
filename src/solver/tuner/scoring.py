from collections import Counter
from dataclasses import asdict, dataclass
from typing import Self

import networkx as nx
import numpy as np

import src.config.pulser as global_conf


@dataclass
class ResultScore:
    """Simple dataclass to store the different scores."""

    total: float
    gini_score: float
    sum_score: float

    @property
    def dict(self) -> dict:
        return asdict(self)

    @classmethod
    def null(cls, gini: float) -> Self:
        return cls(total=0, gini_score=gini, sum_score=0)


def score(counts: Counter, graph: nx.Graph) -> ResultScore:
    """Scores the given MIS result, comparing with the given graph's MIS.

    If a result scores below the threshold in the Gini score, the result is voided.

    Args:
        counts (Counter): Result's counter.
        graph (nx.Graph): Corresponding graph.

    Returns:
        ResultScore: A combination of the different scores.
    """
    gini_score = _gini(counts)

    if gini_score < global_conf.GINI_THRESHOLD:
        return ResultScore.null(gini=gini_score)

    sum_score = _sum_score(counts, graph)

    final_score = sum_score * gini_score

    return ResultScore(total=final_score, gini_score=gini_score, sum_score=sum_score)


def _gini(counts: Counter) -> float:
    """Returns the Gini coefficient of a counter instance."""
    values = list(counts.values())

    # Mean absolute difference
    mad = np.abs(np.subtract.outer(values, values)).mean()
    # Relative mean absolute difference
    rmad = mad / np.mean(values)
    # Gini coefficient
    g = 0.5 * rmad
    return g


def _is_independent_set(bitstring: np.ndarray, graph: nx.Graph) -> bool:
    selected_nodes = [node for node, bit in enumerate(bitstring) if bit == "1"]
    return len(graph.subgraph(selected_nodes).edges) == 0


def _get_configuration_score(configuration: np.ndarray, graph: nx.graph) -> int:
    bitstring = np.array(list(configuration), dtype=int)

    if not _is_independent_set(bitstring, graph):
        return 0

    return sum(bitstring)


def _sum_score(counts: dict, graph: nx.Graph) -> float:
    """Calculates the mean of the independent set sizes, weighted by their count."""
    total_score = sum(
        [count * _get_configuration_score(conf, graph) for conf, count in counts.items()],
    )
    return total_score / sum(counts.values())
