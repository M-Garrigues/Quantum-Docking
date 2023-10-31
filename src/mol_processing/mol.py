"""Module to deal with the Rdkit molecules and their features."""

from rdkit.Chem import ChemicalFeatures, Mol

from src.config.general import SELECTED_FEATURES
from src.mol_processing.features import Feature


def get_features(mol: Mol, mol_id: str) -> list[Feature]:
    """Get features from a molecule, from the configuration's selected families.

    Args:
        mol (Mol): Rdkit molecule.
        mol_id (str): The id of the molecule.

    Returns:
        list[Feature]: List of all features selected.
    """
    FACTORY = ChemicalFeatures.BuildFeatureFactory("./data/BaseFeatures.fdef")
    features = []

    for family in SELECTED_FEATURES.keys():
        features += [
            Feature(rd_feat, mol_id)
            for rd_feat in FACTORY.GetFeaturesForMol(mol, includeOnly=family)
        ]

    return features
