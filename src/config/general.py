"""Global configuration module."""

from typing import Final

""" Interactions configuration """
FLEXIBILITY_CONSTANT_TAU: Final[float] = 1
INTERACTION_DISTANCE_EPSILON: Final[float] = 1


SELECTED_FEATURES: Final[dict] = {
    "Donor": {"abbreviation": "d", "attractors": ["a", "h", "n", "p", "ar", "d"], "color": "blue"},
    "Acceptor": {
        "abbreviation": "a",
        "attractors": ["a", "h", "n", "p", "ar", "d"],
        "color": "yellow",
    },
    "NegIonizable": {
        "abbreviation": "n",
        "attractors": ["a", "h", "n", "p", "ar", "d"],
        "color": "red",
    },
    "PosIonizable": {
        "abbreviation": "p",
        "attractors": ["a", "h", "n", "p", "ar", "d"],
        "color": "green",
    },
    "Aromatic": {
        "abbreviation": "ar",
        " attractors": ["a", "h", "n", "p", "ar", "d"],
        "color": "orange",
    },
    "Hydrophobe": {
        "abbreviation": "h",
        "attractors": ["a", "h", "n", "p", "ar", "d"],
        "color": "purple",
    },
}
