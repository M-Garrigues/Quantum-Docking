"""Global configuration module."""

from typing import Final

from mol.features import FeatureFamily

""" Interactions configuration """
FLEXIBILITY_CONSTANT_TAU: Final[float] = 0
INTERACTION_DISTANCE_EPSILON: Final[float] = 0


SELECTED_FEATURES: Final[dict] = {
    "Donor": {"abbreviation": "d", "attractors": ["a"]},
    "Acceptor": {"abbreviation": "a", "attractors": ["d"]},
    "NegIonizable": {"abbreviation": "n", "attractors": ["n"]},
    "PosIonizable": {"abbreviation": "p", "attractors": ["p"]},
    "Aromatic": {"abbreviation": "ar", " attractors": ["ar"]},
    "LumpedHydrophobe": {"abbreviation": "h", "attractors": ["h"]},
}

SELECTED_FEATURE_FAMILIES: Final[list[FeatureFamily]] = [
    FeatureFamily(name, *SELECTED_FEATURES[name]) for name in SELECTED_FEATURES.keys()
]

#     FeatureFamily("Donor", abbreviation="d", attractors=["a"]),
#     FeatureFamily("Acceptor", abbreviation="a", attractors=["d"]),
#     FeatureFamily("NegIonizable", abbreviation="n", attractors=["n"]),
#     FeatureFamily("PosIonizable", abbreviation="p", attractors=["p"]),
#     FeatureFamily("Aromatic", abbreviation="ar", attractors=["ar"]),
#     FeatureFamily("LumpedHydrophobe", abbreviation="h", attractors=["h"]),
# ]
