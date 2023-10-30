from rdkit import Chem

from src.graph.draw import draw_interaction_graph
from src.graph.interaction_graph import build_binding_interaction_graph, build_distance_matrix
from src.mol_processing.features import name_features_by_count
from src.mol_processing.mol import get_features

if __name__ == "__main__":
    receptor = Chem.MolFromPDBFile("data/receptors/1AO2.pdb", sanitize=True, removeHs=False)
    features = get_features(receptor, "receptor")

    f1 = features[:26]
    f2 = features[26:]

    name_features_by_count(f1, is_ligand=True)
    name_features_by_count(f2, is_ligand=False)
    print(features)
    print(len(features))

    L_distance_matrix = build_distance_matrix(f1)
    R_distance_matrix = build_distance_matrix(f2)

    interaction_graph = build_binding_interaction_graph(L_distance_matrix, R_distance_matrix)
    print(interaction_graph)
    draw_interaction_graph(interaction_graph)
