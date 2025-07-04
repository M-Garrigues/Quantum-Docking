"""Module to handle the features, with dedicated objects."""

from collections import Counter
from dataclasses import dataclass
import json
import os
from rdkit.Chem.rdMolChemicalFeatures import MolChemicalFeature
from rdkit import Chem
from rdkit.Chem import ChemicalFeatures
from rdkit.Chem.ChemicalFeatures import MolChemicalFeature
from rdkit import RDConfig
from rdkit.Chem import AllChem
import itertools
from src.config.general import SELECTED_FEATURES
import numpy as np

from src.utils.dataclasses import OrderedTupleDict, TwoWayTuple


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


FAMILY_NAMES_CONVERSION = {
    "a": "Aromatic",
    "P": "PosIonizable",
    "D": "Donor",
    "A": "Acceptor",
    "N": "NegIonizable",
    "H": "Hydrophobe",
}


class Feature:
    """
    A data class for a pharmacophore feature
    """

    def __init__(
        self,
        family: FeatureFamily,
        position: tuple,
        atom_ids: tuple | None = None,
        molecule_id: str | None = None,
    ):
        self.family: FeatureFamily = family
        self.position: np.ndarray = np.array(position)
        self.atom_ids: tuple[int, ...] = atom_ids
        self.molecule_id: str | None = molecule_id
        self.name: str | None = None

    @classmethod
    def from_rdkit(cls, rd_feature: MolChemicalFeature, molecule_id: str):
        family = FeatureFamily.from_family_name(rd_feature.GetFamily())
        return cls(
            family=family,
            position=rd_feature.GetPos(),
            atom_ids=rd_feature.GetAtomIds(),
            molecule_id=molecule_id,
        )

    def __repr__(self) -> str:
        return f"Feature({self.name})"

    def __hash__(self):
        return hash((self.name, self.molecule_id))

    def __eq__(self, other):
        return (self.name, self.position) == (other.name, other.position)

    def __lt__(self, other):
        return str(self) < str(other)

    def __gt__(self, other):
        return str(self) > str(other)

    def __str__(self):
        return self.name if self.name else "Undefined yet"


@dataclass
class MinMaxDistance:
    """
    A data class to store min/max distance information between a feature pair.
    """

    min_dist: float = float("inf")
    min_conf_id: int = -1
    max_dist: float = float("-inf")
    max_conf_id: int = -1


def extract_pharmacophores(
    molecule: Chem.Mol, molecule_id: str, confId: int = -1
) -> list[Feature]:
    """
    Extracts pharmacophore points, ignoring any family not defined
    in the SELECTED_FEATURES dictionary. Assigns names in the format:
    'molecule_id_family.abbreviation_number'.
    """
    if molecule.GetNumConformers() == 0:
        raise ValueError("Input molecule must have a 3D conformation.")

    fdef_name = os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
    factory = ChemicalFeatures.BuildFeatureFactory(fdef_name)
    rdkit_features = factory.GetFeaturesForMol(molecule, confId=confId)

    feature_list = []
    family_counts = {}

    for rd_feat in rdkit_features:
        family_name = rd_feat.GetFamily()
        if family_name not in SELECTED_FEATURES:
            continue
        feature = Feature.from_rdkit(rd_feat, molecule_id)

        if feature:
            family_name = feature.family.name
            abbreviation = feature.family.abbreviation

            count = family_counts.get(family_name, 0)
            family_counts[family_name] = count + 1

            feature.name = f"{molecule_id}_{abbreviation}_{count}"
            feature_list.append(feature)

    return feature_list


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
    """Returns a Feature by name if it exists in a list, else None"""
    for feat in features_list:
        if name == feat.name:
            return feat

    return None


def load_features_from_pma_file(path: str, reversed=True) -> list[Feature]:

    with open(path, "r") as f:
        points_dict = json.loads(f.read())

    points = points_dict["feature_coords"]
    del points_dict

    points = [
        Feature(
            FeatureFamily.from_family_name(FAMILY_NAMES_CONVERSION[point[0]]), point[1], "test"
        )
        for point in points
    ]

    known_positions = set()
    final_features = []

    if reversed:
        points.reverse()

    for feat in points:
        if tuple(feat.position) in known_positions:
            continue
        known_positions.add(tuple(feat.position))
        final_features.append(feat)

    return final_features


def spatial_selection(features_list: list[Feature], coordinates: list[tuple]) -> list[Feature]:
    """Selects features from a list, if they fit in the n sized coordinates intervals.

    Args:
        features_list (list[Feature]): list of features.
        coordinates (list[tuple]): n tuple of tuples of coordinates, representing an interval.

    Returns:
        list[Feature]: The selected features.
    """
    spatial_selection = []
    for feat in features_list:
        accept = True
        for i, axis_pos in enumerate(feat.position):
            if not (coordinates[i][0] <= axis_pos and axis_pos <= coordinates[i][1]):
                accept = False
                break
        if accept:
            spatial_selection.append(feat)

    return spatial_selection


def find_pharmacophore_distances(
    molecule: Chem.Mol, features: list[Feature]
) -> OrderedTupleDict[MinMaxDistance]:
    """
    Calculates the min/max distances between pre-extracted pharmacophore points
    across all conformers, returning a dictionary of MinMaxDistance objects.
    """
    if molecule.GetNumConformers() == 0:
        raise ValueError("Input molecule must have at least one conformation.")
    if len(features) < 2:
        return {}

    num_total_confs = molecule.GetNumConformers()

    # Initialize the results dictionary with MinMaxDistance objects
    distance_results = OrderedTupleDict()
    for pair in itertools.combinations_with_replacement(features, 2):
        distance_results[pair[0], pair[1]] = MinMaxDistance()

    for conf_id in range(num_total_confs):
        conformer = molecule.GetConformer(conf_id)

        conformer_feature_positions = {}
        for feature in features:
            atom_positions = [conformer.GetAtomPosition(i) for i in feature.atom_ids]
            centroid = np.mean([(p.x, p.y, p.z) for p in atom_positions], axis=0)
            conformer_feature_positions[feature] = centroid

        # Update the MinMaxDistance object attributes directly
        for pair, dist_info in distance_results.items():
            pos1 = conformer_feature_positions[pair[0]]
            pos2 = conformer_feature_positions[pair[1]]
            distance = np.linalg.norm(pos1 - pos2)

            if distance < dist_info.min_dist:
                dist_info.min_dist = distance
                dist_info.min_conf_id = conf_id

            if distance > dist_info.max_dist:
                dist_info.max_dist = distance
                dist_info.max_conf_id = conf_id

    return distance_results
