"""Module to handle the features, with dedicated objects."""

from collections import Counter
from dataclasses import dataclass

from rdkit.Chem.rdMolChemicalFeatures import MolChemicalFeature

from config.general import SELECTED_FEATURES


@dataclass(init=True)
class FeatureFamily:
    """Dataclass for the feature families, for clarity purposes."""

    @classmethod
    def from_family_name(cls, name: str):
        """Return the feature family object from the name."""
        return cls(name, **SELECTED_FEATURES[name])  # type: ignore

    name: str
    abbreviation: str
    attractors: list[str]
    color: str


class Feature:
    """Concentrates a feature's informations. Can be used directly as index in dictionnaries."""

    def __init__(self, rd_feature: MolChemicalFeature, molecule_id: str) -> None:
        self.__name: str | None = None
        self.__family: FeatureFamily = FeatureFamily.from_family_name(rd_feature.GetFamily())
        self.__position = rd_feature.GetPos()
        self.__molecule_id = molecule_id

    @property
    def name(self) -> str | None:
        return self.__name

    @name.setter
    def name(self, new_name: str) -> None:
        self.__name = new_name

    @property
    def family(self) -> FeatureFamily:
        return self.__family

    @property
    def position(self) -> tuple:
        return self.__position

    def __hash__(self):
        return hash((self.__name, self.__position, self.__molecule_id))

    def __eq__(self, other):
        return (self.__name, self.__position) == (other.name, other.position)

    def __lt__(self, other):
        return str(self) < str(other)

    def __gt__(self, other):
        return str(self) > str(other)

    def __str__(self):
        return self.name if self.name else "Undefined yet"

    def __repr__(self) -> str:
        return str(self)


def name_features_by_count(features: list[Feature], is_ligand: bool) -> None:
    """Gives in-place a unique ordered name to each feature of the list, according to their type.

    Args:
        features (list[Feature]): List of features of different types
        is_ligand (bool): If the features' molecule is a ligand, their name will be in upper case.
    """
    counter: Counter[str] = Counter()

    for feature in features:
        abbreviation = feature.family.abbreviation
        if is_ligand:
            abbreviation = abbreviation.upper()
        ordered_id = counter[abbreviation]
        counter[abbreviation] += 1
        feature.name = abbreviation + str(ordered_id)


def find_feature_by_name(name: str, features_list: list[Feature]) -> Feature | None:
    for feat in features_list:
        if name == feat.name:
            return feat

    return None
