import copy

import matplotlib.pyplot as plt

from graph.interaction_graph import InteractionNode
from mol_processing.features import Feature
from utils.dataclasses import OrderedTupleDict


def get_skeleton_edges(features: list[Feature], distance_matrix: OrderedTupleDict) -> list[tuple]:
    def _find_group(list_of_lists, element):
        found_index = None
        for index, sublist in enumerate(list_of_lists):
            if element in sublist:
                found_index = index
                break
        return found_index

    edges = set()
    features_remaining = copy.copy(features)
    ordered_distances = sorted(distance_matrix.items(), key=lambda x: x[1])
    groups = [[feature] for feature in features_remaining]

    while len(features_remaining) > 0 or len(groups) > 1:
        smallest_edge = ordered_distances[0]
        ordered_distances.pop(0)
        new_edge = smallest_edge[0]
        edges.add(new_edge)

        for feat in new_edge:
            try:
                features_remaining.remove(feat)
            except:
                pass

        first_group_index = _find_group(groups, new_edge[0])
        second_group_index = _find_group(groups, new_edge[1])

        if first_group_index != second_group_index:
            groups[first_group_index] += groups[second_group_index]
            groups.pop(second_group_index)

    return [*(zip(node_a.position, node_b.position) for node_a, node_b in edges)]  # type: ignore


def plot_feat_and_edges(
    features: list[Feature],
    edges: list,
    ax,
    vertical_offset: float = 0,
) -> None:
    for feat in features:
        x, y, z = feat.position
        ax.scatter(x, y, z + vertical_offset, c=feat.family.color, s=50)
        ax.text(x, y, z + vertical_offset + 0.3, feat.name, fontsize=8, color="black")

    for edge in edges:
        x_vector, y_vector, z_vector = edge
        z_vector = tuple(z + vertical_offset for z in z_vector)
        ax.plot(x_vector, y_vector, z_vector, color="tab:gray", linewidth=1)


def _format_axes(ax):
    """Visualization options for the 3D axes."""
    # Turn gridlines off
    ax.grid(False)
    ax.set_axis_off()

    # Suppress tick labels
    for dim in (ax.xaxis, ax.yaxis, ax.zaxis):
        dim.set_ticks([])


def draw_feature_list(features: list[Feature], distance_matrix: OrderedTupleDict) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    edges = get_skeleton_edges(features, distance_matrix)

    plot_feat_and_edges(features, edges, ax)

    _format_axes(ax)
    fig.tight_layout()
    plt.show()


def draw_docking(
    L_features: list[Feature],
    L_distance_matrix: OrderedTupleDict,
    R_features: list[Feature],
    R_distance_matrix: OrderedTupleDict,
    interacting_nodes: list[InteractionNode],
) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    vertical_offset = 20

    R_edges = get_skeleton_edges(R_features, R_distance_matrix)
    plot_feat_and_edges(R_features, R_edges, ax)

    L_edges = get_skeleton_edges(L_features, L_distance_matrix)
    plot_feat_and_edges(L_features, L_edges, ax, vertical_offset=vertical_offset)

    for interaction in interacting_nodes:
        x_vector, y_vector, z_vector = zip(
            interaction.R_feature.position,
            interaction.L_feature.position,
        )
        R_z_vector, L_z_vector = z_vector
        L_z_vector += vertical_offset
        z_vector = (R_z_vector, L_z_vector)
        ax.plot(
            x_vector,
            y_vector,
            z_vector,
            color="tab:green",
            linewidth=3,
            linestyle="dashed",
        )

    _format_axes(ax)
    fig.tight_layout()
    plt.show()
