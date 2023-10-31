"""Module for plotting the graphs."""

import matplotlib.pyplot as plt
import networkx as nx


def draw_interaction_graph(G: nx.Graph) -> None:
    """Draws the weighted interaction graph.

    Args:
        G (nx.Graph): Weighted nodes undirected graph.
    """
    seed = 42  # Seed random number generators for reproducibility

    # Specify the positions of nodes (you can use different layouts)
    pos = nx.spring_layout(G, scale=10, iterations=10, k=1, seed=seed)

    # Draw the graph
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_size=[G.nodes[n]["weight"] * 100 for n in G.nodes()],
        node_color="blue",
        font_size=7,
    )
    nx.draw_networkx_edge_labels(G, pos, edge_labels={(u, v): "" for u, v in G.edges()})
    nx.draw_networkx_edges(G, pos, edge_color="black", width=0.2)

    # Display the graph
    plt.axis("off")
    plt.show()
