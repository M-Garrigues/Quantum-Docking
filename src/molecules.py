import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from rdkit import Chem
from rdkit.Chem import MACCSkeys


def extract_pharmacophore_from_pdb(pdb_file: str) -> MACCSkeys:
    """Extracts the 2D pharmacophore from a PDB file and displays the molecule.

    Args:
        pdb_file (str): The path to the PDB file.

    Returns:
        MACCSkeys: The 2D pharmacophore representation.
    """
    mol = Chem.MolFromPDBFile(pdb_file)

    if mol is None:
        print("Failed to load molecule from PDB file.")
        return

    pharmacophore = MACCSkeys.GenMACCSKeys(mol)
    print(pharmacophore)
    # img = Draw.Pharm3DToImage(pharmacophore)
    # img.show()

    return pharmacophore


def extract_pairwise_matrix(pharmacophore: MACCSkeys) -> np.ndarray:
    """Extracts the labeled pairwise matrix from a 2D pharmacophore representation.

    Args:
        pharmacophore (MACCSkeys): The 2D pharmacophore representation.

    Returns:
        np.ndarray: The labeled pairwise matrix.
    """
    matrix = np.zeros((pharmacophore.GetNumBits(), pharmacophore.GetNumBits()))

    for i in range(pharmacophore.GetNumBits()):
        for j in range(i, pharmacophore.GetNumBits()):
            if i != j:
                matrix[i, j] = matrix[j, i] = pharmacophore.GetBit(i) & pharmacophore.GetBit(j)

    return matrix


def construct_binding_interaction_graph(
    matrix_ligand: np.ndarray,
    matrix_receptor: np.ndarray,
) -> nx.Graph:
    """Constructs a binding interaction graph between a ligand and receptor based on their matrices.

    Args:
        matrix_ligand (np.ndarray): The labeled pairwise matrix for the ligand.
        matrix_receptor (np.ndarray): The labeled pairwise matrix for the receptor.

    Returns:
        nx.Graph: The binding interaction graph.
    """
    binding_matrix = matrix_ligand.dot(matrix_receptor.T)
    binding_graph = nx.Graph(binding_matrix)

    return binding_graph


def display_interaction_graph(binding_graph: nx.Graph):
    """Displays the binding interaction graph using Matplotlib.

    Args:
        binding_graph (nx.Graph): The binding interaction graph.
    """
    pos = nx.spring_layout(binding_graph)  # Position nodes using the spring layout

    plt.figure(figsize=(8, 8))
    nx.draw(
        binding_graph,
        pos,
        with_labels=True,
        node_size=200,
        node_color="lightblue",
        font_size=10,
    )
    edge_labels = {(u, v): d["weight"] for u, v, d in binding_graph.edges(data=True)}
    nx.draw_networkx_edge_labels(binding_graph, pos, edge_labels=edge_labels)

    plt.title("Binding Interaction Graph")
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    pdb_file = "./data/receptors/1AO2.pdb"  # Replace with the path to your PDB file
    receptor_pharmacophore = extract_pharmacophore_from_pdb(pdb_file)
    receptor_matrix = extract_pairwise_matrix(receptor_pharmacophore)

    pdb_file = "./data/ligands/H2O2.pdb"  # Replace with the path to your PDB file
    ligand_pharmacophore = extract_pharmacophore_from_pdb(pdb_file)
    ligand_matrix = extract_pairwise_matrix(ligand_pharmacophore)

    # Construct the binding interaction graph
    binding_graph = construct_binding_interaction_graph(ligand_matrix, receptor_matrix)

    # You can analyze or visualize the binding interaction graph as needed
    display_interaction_graph(binding_graph)
