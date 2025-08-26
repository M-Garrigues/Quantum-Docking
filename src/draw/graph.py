"""Module for plotting the graphs."""

import math
from collections.abc import Mapping
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.figure import Figure


def draw_interaction_graph(G: nx.Graph, highlight_nodes: list[str] | None = None) -> None:
    """Draws the weighted interaction graph.

    Args:
        G (nx.Graph): Weighted nodes undirected graph.
        highlight_nodes (list[str] | None): Nodes to highlight in red.
    """
    if not highlight_nodes:
        highlight_nodes = []
    seed = 42  # Seed random number generators for reproducibility

    # Specify the positions of nodes (you can use different layouts)
    pos = nx.spring_layout(G, scale=3, iterations=10, k=1, seed=seed)

    if any("weight" in G.nodes[node] for node in G.nodes):
        nodes_size = [math.sqrt(1 + G.nodes[n]["weight"]) * 1e3 for n in G.nodes()]
    else:
        nodes_size = [300 for _ in G.nodes()]

    # Draw the graph
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_size=nodes_size,
        node_color=["tab:red" if n in highlight_nodes else "tab:blue" for n in G],
        font_size=7,
    )
    nx.draw_networkx_edges(G, pos, edge_color="black", width=0.2)

    # Display the graph
    plt.axis("off")
    plt.show()


def draw_multiple_cliques(G: nx.Graph, cliques: list[list[str]]) -> None:
    """Draw multiple cliques in consecutive graphs."""
    for clique in cliques:
        draw_interaction_graph(G, clique)


def plot_weighted_graph(
    graph: nx.Graph,
    pos: Mapping[Any, tuple[float, float]] | None = None,
    figsize: tuple[int, int] = (14, 14),
    node_size_range: tuple[int, int] = (20, 300),
    node_weight_attr: str = "weight",
    node_cmap: str = "viridis",
    edge_color: str = "lightgray",
    edge_alpha: float = 0.2,
    layout: str = "spring",
    show_axis: bool = False,
    draw_labels: bool = False,
) -> Figure:
    """
    Plot a graph where nodes have weights and edges are unweighted.
    Node size and color represent node weights.

    Args:
        graph (nx.Graph): NetworkX graph with node weights.
        pos (Mapping[Any, tuple[float, float]] | None): Optional node positions.
        figsize (tuple[int, int]): Figure size.
        node_size_range (tuple[int, int]): Min and max node sizes.
        node_weight_attr (str): Attribute name for node weights.
        node_cmap (str): Matplotlib colormap name.
        edge_color (str): Color of edges.
        edge_alpha (float): Transparency of edges.
        layout (str): Layout to compute positions ('spring', 'kamada_kawai', etc.).
        show_axis (bool): Whether to show the axis.
        draw_labels (bool): Whether to draw node labels.

    Returns:
        matplotlib.figure.Figure: The plotted figure.
    """
    if pos is None:
        match layout:
            case "spring":
                pos = nx.spring_layout(graph, seed=42)
            case "kamada_kawai":
                pos = nx.kamada_kawai_layout(graph)
            case "spectral":
                pos = nx.spectral_layout(graph)
            case "circular":
                pos = nx.circular_layout(graph)
            case _:
                raise ValueError(f"Unsupported layout: {layout}")

    # Extract node weights
    node_weights_raw = np.array([graph.nodes[n].get(node_weight_attr, 1.0) for n in graph.nodes])
    # Normalize node sizes and colors
    min_w, max_w = node_weights_raw.min(), node_weights_raw.max()
    if min_w == max_w:
        norm_weights = np.full_like(node_weights_raw, 0.5)
    else:
        norm_weights = (node_weights_raw - min_w) / (max_w - min_w)

    node_sizes = node_size_range[0] + norm_weights * (node_size_range[1] - node_size_range[0])

    fig, ax = plt.subplots(figsize=figsize)
    nx.draw_networkx_edges(
        graph,
        pos,
        ax=ax,
        edge_color=edge_color,
        alpha=edge_alpha,
        width=0.5,
    )
    nodes = nx.draw_networkx_nodes(
        graph,
        pos,
        ax=ax,
        node_size=node_sizes,
        node_color=norm_weights,
        cmap=plt.get_cmap(node_cmap),
        alpha=0.9,
    )

    if draw_labels:
        nx.draw_networkx_labels(graph, pos, ax=ax, font_size=8)

    if not show_axis:
        ax.set_axis_off()
    plt.colorbar(nodes, ax=ax, shrink=0.8, label="Normalised node weight")
    plt.tight_layout()
    return fig
