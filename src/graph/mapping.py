"""Module containing functions to map the MC problem to a UDG-MIS."""

import networkx as nx
import numpy as np
from pulser import Register
from pulser.devices import Chadoq2
from scipy.optimize import minimize
from scipy.spatial import KDTree
from scipy.spatial.distance import pdist, squareform


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
        options={"maxiter": 200000},
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
