from collections import Counter

from attr import dataclass
from rdkit.Chem.rdMolChemicalFeatures import MolChemicalFeature

from src.config import SELECTED_FEATURES


@dataclass(init=True)
class FeatureFamily:
    @classmethod
    def from_family_name(cls, name: str):
        return cls(name, **SELECTED_FEATURES[name])  # type: ignore

    name: str
    abbreviation: str
    attractors: list[str]


class Feature:
    def __init__(self, rd_feature: MolChemicalFeature, molecule_id: str) -> None:
        self.__name: str | None = None
        self.__family: FeatureFamily = rd_feature.GetFamily()
        self.__position = rd_feature.GetPos()
        self.__molecule_id = molecule_id

    @property
    def name(self):
        return self.__name

    @name.setter
    def name(self, new_name: str) -> None:
        self.__name = new_name

    @property
    def family(self):
        return self.__family

    def __print__(self):
        print(f"Feature {self.__name} - {self.__family} - Coordinated {self.__position}")


def name_features_by_count(features: list[Feature], is_ligand: bool) -> None:
    """Gives in-place a unique ordered name to each feature of the list, according to their type.

    Args:
        features (list[Feature]): List of features of different types
        is_ligand (bool): If the features' molecule is a ligand, their name will be in upper case.
    """
    counter: Counter[int] = Counter()

    for feature in features:
        abbreviation = feature.family.abbreviation
        ordered_id = counter[abbreviation]
        counter[abbreviation] += 1
        feature.name = abbreviation + str(ordered_id)
