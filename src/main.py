from pprint import pprint

from rdkit import Chem

from src.mol_processing.features import name_features_by_count
from src.mol_processing.interaction_graph import build_distance_matrix
from src.mol_processing.mol import get_features

if __name__ == "__main__":
    receptor = Chem.MolFromPDBFile("data/receptors/1AO2.pdb", sanitize=True, removeHs=False)
    features = get_features(receptor, "receptor")

    name_features_by_count(features, is_ligand=True)
    print(features)
    print(len(features))

    distance_matrix = build_distance_matrix(features)

    pprint(distance_matrix)
