import networkx as nx


def find_max_clique(G: nx.Graph) -> list:
    cliques = list(nx.find_cliques(G))
    largest_clique = max(cliques, key=len)
    return largest_clique
