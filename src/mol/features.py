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

    def __print__(self):
        print(f"Feature {self.__name} - {self.__family} - Coordinated {self.__position}")
