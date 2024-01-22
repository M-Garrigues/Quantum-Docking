import networkx as nx
import pulser
from matplotlib import pyplot as plt
from networkx.algorithms import approximation

from src.solver.opt_vqaa import VQAA
from src.solver.utils.graph_register import GraphRegister


def plot_distribution(counts, actual_solution):
    counts = dict(sorted(counts.items(), key=lambda item: item[1], reverse=True))
    color_dict = {key: "r" if key in actual_solution else "g" for key in counts}
    plt.figure(figsize=(12, 6))
    plt.xlabel("bitstrings")
    plt.ylabel("counts")
    plt.bar(counts.keys(), counts.values(), width=0.5, color=color_dict.values())
    plt.xticks(rotation="vertical")
    plt.show()


def full_test(path):
    register = GraphRegister.from_json(path)

    graph = register.graph

    I = approximation.maximum_independent_set(graph)
    num_nodes = graph.number_of_nodes()
    bitstring = "".join("1" if node in I else "0" for node in range(num_nodes))

    print(f"Maximum independent set of G: {I}")
    register.draw(
        blockade_radius=pulser.devices.Chadoq2.rydberg_blockade_radius(1.0),
        draw_graph=True,
        draw_half_radius=True,
    )
    pos = nx.spring_layout(graph)
    nx.draw(
        graph,
        pos=pos,
        with_labels=True,
        node_color=["tab:red" if n in I else "tab:blue" for n in graph],
    )

    counts, score, _ = VQAA(path, optimisation_rounds=30, store_results=True)

    print(score)

    plot_distribution(counts, bitstring)


if __name__ == "__main__":
    full_test("./data/registers/9/9_mesh.json")
