"""Molecular Docking main module."""
from rdkit import Chem

from draw.features import draw_docking
from graph.interaction_graph import (
    InteractionNode,
    build_binding_interaction_graph,
    build_distance_matrix,
)
from mol_processing.features import name_features_by_count
from mol_processing.mol import get_features

if __name__ == "__main__":
    mol = Chem.MolFromPDBFile("../data/receptors/1AO2.pdb", sanitize=True, removeHs=False)
    features = get_features(mol, "receptor")[::3]

    R_features = features
    L_features = get_features(mol, "ligand")[::5]

    name_features_by_count(R_features, is_ligand=False)
    name_features_by_count(L_features, is_ligand=True)

    R_distance_matrix = build_distance_matrix(R_features)
    L_distance_matrix = build_distance_matrix(L_features)

    # draw_feature_list(R_features, R_distance_matrix)

    interaction_graph = build_binding_interaction_graph(L_distance_matrix, R_distance_matrix)
    # draw_interaction_graph(interaction_graph)

    docking = [
        InteractionNode(R_feature=R_features[0], L_feature=L_features[1], weight=10),
        InteractionNode(R_feature=R_features[8], L_feature=L_features[5], weight=10),
        InteractionNode(R_feature=R_features[12], L_feature=L_features[6], weight=10),
    ]

    draw_docking(
        L_features,
        L_distance_matrix,
        R_features,
        R_distance_matrix,
        interacting_nodes=docking,
    )
    # c = find_max_clique(interaction_graph)
    # print(c)
    # draw_interaction_graph(c)
