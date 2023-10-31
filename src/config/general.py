"""Global configuration module."""

from typing import Final

# from src.mol_processing.features import FeatureFamily

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
