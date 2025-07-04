"""Global configuration module."""

from typing import Final

# Interactions configuration

INTERACTION_DISTANCE_EPSILON: Final[float] = 1


SELECTED_FEATURES: Final[dict] = {
    "Donor": {"abbreviation": "d", "attractors": ["a"], "color": "blue"},
    "Acceptor": {
        "abbreviation": "a",
        "attractors": ["d"],
        "color": "red",
    },
    "NegIonizable": {
        "abbreviation": "n",
        "attractors": ["p"],
        "color": "yellow",
    },
    "PosIonizable": {
        "abbreviation": "p",
        "attractors": ["n"],
        "color": "green",
    },
    "Aromatic": {
        "abbreviation": "ar",
        "attractors": ["ar"],
        "color": "orange",
    },
    "Hydrophobe": {
        "abbreviation": "h",
        "attractors": ["h"],
        "color": "purple",
    },
}
