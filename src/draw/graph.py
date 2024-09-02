"""Module for plotting the graphs."""

import math
import matplotlib.pyplot as plt
import networkx as nx


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
