"""Module containing functions to map the MC problem to a UDG-MIS."""

import networkx as nx
import numpy as np
from pulser import Register
from pulser.devices import Chadoq2
from scipy.optimize import minimize
from scipy.spatial import KDTree
from scipy.spatial.distance import pdist, squareform

from graph.interaction_graph import InteractionNode
from mol_processing.features import Feature, find_feature_by_name


def build_complementary_graph(G: nx.graph) -> nx.graph:
    """Returns the complementary graph of the given graph."""
    return nx.complement(G)


def _evaluate_mapping(new_coords, *args):
    """Cost function to minimize. Ideally, the pairwise distances are conserved."""
    test, shape = args
    new_coords = np.reshape(new_coords, shape)
    new_Q = squareform(Chadoq2.interaction_coeff / pdist(new_coords) ** 6)
    return np.linalg.norm(new_Q - test)


def map_to_UDG(G: nx.graph) -> dict:
    """Maps a given graph to a UDG compatible position.
       This is a heuristic, there is no guarantee that the position is reached or even exists.

    Args:
        G (nx.graph): Networkx graph.

    Returns:
        dict: A dictionnary containing {node: position} pairs.
    """
    adjacency_matrix = nx.adjacency_matrix(G).todense()
    shape = (len(adjacency_matrix), 2)
    np.random.seed(0)
    x0 = np.random.random(shape).flatten()
    res = minimize(
        _evaluate_mapping,
        x0,
        args=(adjacency_matrix * 2, shape),
        method="COBYLA",
        tol=1e-6,
        options={"maxiter": 2e5},
    )
    coords = np.reshape(res.x, (len(adjacency_matrix), 2))

    # From https://github.com/pasqal-io/Pulser/blob/a531a500f693838751934e5ec0e26b37caaf7f4f/pulser-core/pulser/register/_reg_drawer.py#L214 # pylint: disable=C0301
    epsilon = 1e-9  # Accounts for rounding errors
    edges = KDTree(coords).query_pairs(Chadoq2.rydberg_blockade_radius(1.0) * (1 + epsilon))
    bonds = coords[(tuple(edges),)]

    if len(bonds) != len(adjacency_matrix):
        print(
            "Problem while generating UDG graph: the number of resulting edges is not equal to the inital number of  edges.",  # pylint: disable=C0301
        )

    return dict(zip(G.nodes(), coords))


def add_quantum_link(G: nx.graph, node_A: str, node_B: str, chain_size: int = 1) -> None:
    link_name = "LINK"
    n_nodes = 2 * chain_size  # To keep the entanglement in the desired state
    # Reference: https://journals.aps.org/prxquantum/pdf/10.1103/PRXQuantum.3.030305

    for n in range(n_nodes):
        G.add_node(link_name + "-" + str(n))

        if n == 0:
            G.add_edge(node_A, link_name + "-" + str(n))
        else:
            G.add_edge(link_name + "-" + str(n - 1), link_name + "-" + str(n))

        if n == n_nodes - 1:
            G.add_edge(link_name + "-" + str(n), node_B)

    G.remove_edge(node_A, node_B)


def demo_positions(G: nx.graph) -> dict:
    return {
        "H28-d6": (-4.2, 16.3),
        "H12-d6": (-3.3, 4.6),
        "H30-d6": (6.4, 11.2),
        "H12-d8": (-0.1, -6.8),
        "H30-d8": (11.7, -7.3),
        "H28-d8": (5.3, -17.3),
        "LINK-1": (14, 8),
        "LINK-2": (15, 0),
    }


def embed_to_register(positions: dict) -> Register:
    """Creates a register from a dict of nodes and there position."""
    # qubits = {node.name: position for node, position in positions.items()}
    return Register(positions)


def embed_problem_to_QPU(G: nx.Graph) -> Register:
    """Embeds a MC solving problem to a QPU register, transforming it to a UFG-MIS problem.

    Args:
        G (nx.Graph): A networkx graph corresponding to a MC problem.

    Returns:
        Register: _description_
    """
    # Mapping from a MC to a MIS problem.
    complementary = build_complementary_graph(G)

    # Map the MIS problem graph to a corresponding UDG graph.
    UDG_positions = map_to_UDG(complementary)

    # Return the UDG graph embedded into the register.
    return embed_to_register(UDG_positions)


def results_to_interaction_graph_cliques(
    cliques: list[list[str]],
    L_features: list[Feature],
    R_features: list[Feature],
) -> list[list[InteractionNode]]:
    """Maps back the results of the quantum solver to the corresponding interaction nodes.

    Args:
        cliques (list[list[str]]):
            List of cliques, which are a list of the corresponding nodes' names.
        R_features (list[Feature]): Receptor features list.
        L_features (list[Feature]): Ligand features list.

    Returns:
        list[list[InteractionNode]]: _description_
    """
    cliques_list = []

    for clique in cliques:
        nodes_list = []
        for node_name in clique:
            L_feat_name, R_feat_name = tuple(node_name.split("-"))

            L_feat = find_feature_by_name(L_feat_name, L_features)
            R_feat = find_feature_by_name(R_feat_name, R_features)

            nodes_list.append(InteractionNode(L_feat, R_feat, 1))

        cliques_list.append(nodes_list)

    return cliques_list
