from math import sqrt

import numpy as np

from mol.features import Feature


def euclidean_distance(p1: tuple, p2: tuple) -> float:
    """Calculates the euclidean distance between 2 points with len(p) coordinates."""
    assert len(p1) == len(p2)
    return round(sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2))), 1)


def build_distance_matrix(features: list[Feature]) -> np.typing.NDArray[np.float64]:
    """Generates a pairwise upper triangular distance matrix between features.

    Args:
        features (list[Feature]): List of features.

    Returns:
        np.array: Upper triangular distance matrix.
    """
    distance_matrix = np.zeros((len(features), len(features)))

    for i, feat_1 in enumerate(features):
        for j, feat_2 in enumerate(features, start=i):
            distance = euclidean_distance(feat_1.position, feat_2.position)
            distance_matrix[i][j] = distance

    return distance_matrix
