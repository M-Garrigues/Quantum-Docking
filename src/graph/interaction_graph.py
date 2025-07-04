"""Module containing all the functions needed to build a binding interaction graph."""

from dataclasses import dataclass
import itertools

import networkx as nx

from src.config.general import INTERACTION_DISTANCE_EPSILON
from src.config.potentials import POTENTIAL_FUNCTION
from src.mol_processing.features import Feature, MinMaxDistance
from src.utils.dataclasses import OrderedTupleDict
from src.utils.distances import euclidean_distance


@dataclass
class InteractionNode:
    """Clarity class to have a node's informations and its name calculated."""

    L_feature: Feature
    R_feature: Feature
    weight: float

    @property
    def name(self):
        return f"{self.L_feature.name}-{self.R_feature.name}"

    def __hash__(self) -> int:
        return hash((self.__class__.name, self.name, self.weight))


def build_distance_matrix(
    features: list[Feature],
) -> OrderedTupleDict[float]:
    """Generates a pairwise upper triangular distance matrix between features.

    Args:
        features (list[Feature]): List of features.

    Returns:
        OrderedTupleDict[tuple[Feature, Feature], float]:
            dict with a tuple of features as index containing their pairwise distance.
    """
    distance_matrix: OrderedTupleDict[float] = OrderedTupleDict()

    for i, feat_1 in enumerate(features):
        for feat_2 in features[i:]:
            distance_matrix[feat_1, feat_2] = euclidean_distance(feat_1.position, feat_2.position)

    return distance_matrix


def filter_receptor_features_close_to_ligand(
    L_features: list[Feature], R_features: list[Feature], max_distance: float = 4
) -> tuple[list[Feature], list[Feature]]:
    selected_L_features = set()
    selected_R_features = set()

    for L_feat in L_features:
        for R_feat in R_features:

            distance = euclidean_distance(L_feat.position, R_feat.position)

            if distance < max_distance and features_are_attracted(L_feat, R_feat) and distance > 0:

                selected_L_features.add(L_feat)
                selected_R_features.add(R_feat)

    return list(selected_L_features), list(selected_R_features)


def pairs_distances_match(L_feature_min_max: MinMaxDistance, R_features_distance: float) -> bool:
    """Determines whether two features pairs in different molecules can simultaneously interact with
        each other, based on distance between features, the variation of distance due to the ligand's
        flexibility and the Epsilon interaction distance parameter.

    Args:
        L_feature_min_max (MinMaxDistance): Pair min/max distances between features in ligand molecule.
        R_features_distance (float): Pair distance between features in receptor molecule.

    Returns:
        bool: Wether the pairs can interact.
    """
    return (
        L_feature_min_max.min_dist - (2 * INTERACTION_DISTANCE_EPSILON)
        <= R_features_distance
        <= L_feature_min_max.max_dist + (2 * INTERACTION_DISTANCE_EPSILON)
    )


def features_are_attracted(L_feature: Feature, R_feature: Feature) -> bool:
    """Determines wether two features will interact based on their pharmacophoric family.

    Args:
        L_feature (Feature): Ligand feature
        R_feature (Feature): Receptor feature

    Returns:
        bool: True if they are attracted.
    """
    return (
        L_feature.family.abbreviation in R_feature.family.attractors
        and R_feature.family.abbreviation in L_feature.family.attractors
    )


def find_possible_edges(
    L_feature_pair: tuple[Feature, Feature],
    R_feature_pair: tuple[Feature, Feature],
) -> list[tuple[tuple[Feature, Feature], tuple[Feature, Feature]]]:
    """For a given pair of edges in the Ligand and the Receptor, returns the physically possible
       edges that will be integrated in the binding interaction graph.

    Args:
        L_feature_pair (tuple[Feature, Feature]): Ligand feature pair.
        R_feature_pair (tuple[Feature, Feature]): Receptor feature pair.

    Returns:
        list[tuple[tuple[Feature, Feature], tuple[Feature, Feature]]]:
            A list containing at max two possible edges of the interaction graph.
    """
    possible_edges: list[tuple[tuple[Feature, Feature], tuple[Feature, Feature]]] = []

    edge_interactions = (
        (L_feature_pair[0], R_feature_pair[1]),
        (L_feature_pair[1], R_feature_pair[0]),
    )
    edge_interactions_reversed = (
        (L_feature_pair[1], R_feature_pair[1]),
        (L_feature_pair[0], R_feature_pair[0]),
    )

    for interaction_nodes in [
        edge_interactions,
        edge_interactions_reversed,
    ]:
        possible_edges.append(interaction_nodes)

    return possible_edges


def build_nx_weighted_graph(
    edges: set[tuple[InteractionNode, InteractionNode]],
    nodes: set[InteractionNode],
) -> nx.Graph:
    """Builds a networkx graph object from a set of nodes and edges.

    Args:
        edges (set[tuple[InteractionNode, InteractionNode]]):
            Set of edges, tuple of InteractionNode objects.
        nodes (set[InteractionNode]): Set of InteractionNode objects.

    Returns:
        nx.Graph: Created networkx graph object.
    """
    G = nx.Graph()

    # Add nodes to the graph with their weights
    for node in nodes:
        G.add_node(node.name, weight=node.weight, interaction_node=node)

    # Add edges to the graph
    for edge in edges:
        G.add_edge(edge[0].name, edge[1].name)

    return G


def build_binding_interaction_graph(
    L_distance_matrix: OrderedTupleDict[MinMaxDistance],
    R_distance_matrix: OrderedTupleDict[float],
) -> nx.Graph:
    """Given the distance matrixes for two molecules and their families,
       build a full binding interaction graph.

    Args:
        L_distance_matrix (OrderedTupleDict[MinMaxDistance]): Ligand's distance matrix.
        R_distance_matrix (OrderedTupleDict[float]): Receptor's distance matrix.

    Returns:
        nx.Graph: Full binding graph object.
    """
    possible_edges: list[tuple[tuple[Feature, Feature], tuple[Feature, Feature]]] = []

    for L_pair, L_min_max in L_distance_matrix.items():
        for R_pair, R_distance in R_distance_matrix.items():
            if not pairs_distances_match(L_min_max, R_distance):
                continue

            possible_edges += find_possible_edges(L_pair, R_pair)  # type: ignore

    nodes: set[InteractionNode] = set()
    edges: set[tuple[InteractionNode, InteractionNode]] = set()

    for edge in possible_edges:
        edge_nodes: set[InteractionNode] = set()
        for node in edge:
            edge_nodes.add(
                InteractionNode(
                    L_feature=node[0],
                    R_feature=node[1],
                    weight=POTENTIAL_FUNCTION[
                        node[0].family.abbreviation,
                        node[1].family.abbreviation,
                    ],
                ),
            )

        if len(edge_nodes) == 1:  # The nodes are both the same. Avoids self loop in the graph.
            continue

        edges.add(tuple(edge_nodes))  # type: ignore
        nodes.update(edge_nodes)

    return build_nx_weighted_graph(edges, nodes)


def build_weighted_binding_interaction_graph(
    L_features,
    R_features,
    L_distance_matrix: OrderedTupleDict[MinMaxDistance],
    R_distance_matrix: OrderedTupleDict[float],
) -> nx.Graph:

    nodes = [
        InteractionNode(
            L_feature=L_feat,
            R_feature=R_feat,
            weight=POTENTIAL_FUNCTION[
                L_feat.family.abbreviation,
                R_feat.family.abbreviation,
            ],
        )
        for L_feat, R_feat in itertools.product(L_features, R_features)
    ]

    edges: set[tuple[InteractionNode, InteractionNode]] = set()
    temp: set[tuple[InteractionNode, InteractionNode]] = set()

    for i, node_a in enumerate(nodes):
        for node_b in nodes[i + 1 :]:

            L_min_max = L_distance_matrix[node_a.L_feature, node_b.L_feature]
            R_distance = R_distance_matrix[node_a.R_feature, node_b.R_feature]

            if pairs_distances_match(L_min_max, R_distance):
                edges.add((node_a, node_b))
            temp.add((node_a, node_b))
    return build_nx_weighted_graph(edges, nodes)


def remove_weights(weighted_graph: nx.Graph) -> None:
    for nodes in nx.to_dict_of_dicts(weighted_graph).keys():
        for attrs in nodes.values():
            attrs.pop("weight", None)
