from math import sqrt

def euclidean_distance(p1: tuple, p2: tuple) -> float:
    """Calculates the euclidean distance between 2 points with len(p) coordinates."""
    assert len(p1) == len(p2)
    return round(sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2))), 1)
