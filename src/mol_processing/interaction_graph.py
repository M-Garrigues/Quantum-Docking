from src.graph.utils import euclidean_distance
from src.mol_processing.features import Feature


def build_distance_matrix(features: list[Feature]) -> dict[tuple[Feature, Feature], float]:
    """Generates a pairwise upper triangular distance matrix between features.

    Args:
        features (list[Feature]): List of features.

    Returns:
        dict[float]: dict with a tuple of features as index containing their pairwise distance.
    """
    distance_matrix = {}

    for i, feat_1 in enumerate(features):
        for feat_2 in features[i + 1 :]:
            distance = euclidean_distance(feat_1.position, feat_2.position)
            distance_matrix[feat_1, feat_2] = distance

    return distance_matrix
