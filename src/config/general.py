"""Global configuration module."""

from typing import Final

""" Interactions configuration """
FLEXIBILITY_CONSTANT_TAU: Final[float] = 1
INTERACTION_DISTANCE_EPSILON: Final[float] = 1


SELECTED_FEATURES: Final[dict] = {
    "Donor": {"abbreviation": "d", "attractors": ["a"], "color": "blue"},
    "Acceptor": {"abbreviation": "a", "attractors": ["d"], "color": "yellow"},
    "NegIonizable": {"abbreviation": "n", "attractors": ["n"], "color": "red"},
    "PosIonizable": {"abbreviation": "p", "attractors": ["p"], "color": "green"},
    "Aromatic": {"abbreviation": "ar", " attractors": ["ar"], "color": "orange"},
    "Hydrophobe": {"abbreviation": "h", "attractors": ["h"], "color": "purple"},
}
