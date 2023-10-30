import matplotlib.pyplot as plt
import networkx as nx


def draw_interaction_graph(G: nx.Graph):
    seed = 13648  # Seed random number generators for reproducibility

    # Specify the positions of nodes (you can use different layouts)
    pos = nx.spring_layout(G, scale=1, iterations=10, k=50, seed=seed)

    # Extract weights from the graph
    # weights = [G[u][v]["weight"] for u, v in G.nodes()]

    # Draw the graph
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_size=[G.nodes[n]["weight"] * 50 for n in G.nodes()],
        node_color="lightblue",
        font_size=8,
    )
    nx.draw_networkx_edge_labels(G, pos, edge_labels={(u, v): "" for u, v in G.edges()})
    nx.draw_networkx_edges(G, pos, edge_color="red", width=0.3)

    # Display the graph
    plt.axis("off")
    plt.show()
