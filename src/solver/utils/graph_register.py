import json
import os
from math import sqrt

import networkx as nx
import pulser
from pulser.devices import Chadoq2
from scipy.spatial import KDTree


class GraphRegister(pulser.Register):
    s_qubits: dict = {}
    metadata: dict = {}

    def __init__(self, qubits: dict):
        super().__init__(qubits)
        serialised_dict = {key: list(value) for key, value in qubits.items()}
        self.s_qubits = serialised_dict

    def __repr__(self) -> str:
        return super().__repr__()

    @classmethod
    def from_json(cls, json_path: str) -> pulser.Register:
        with open(json_path) as f:
            json_config = json.load(f)

        q = {name: tuple(coords) for name, coords in json_config["qubits"].items()}
        register = cls(q)

        cls.s_qubits = q

        if "metadata" in json_config.keys():
            register.metadata = json_config["metadata"]

        return register

    def set_metadata(self, new_metadata: dict):
        self.metadata = new_metadata

    def to_json_file(self, json_path: str) -> None:
        register_infos = {"qubits": self.s_qubits, "metadata": self.metadata}

        with open(json_path, mode="w+") as f:
            f.write(json.dumps(register_infos, indent=4))

    @property
    def size(self) -> int:
        return len(self._coords)

    @property
    def graph(self) -> nx.Graph:
        edges = self._find_connected_qubits()
        return nx.Graph(edges)

    def _find_connected_qubits(self) -> list:
        epsilon = 1e-9
        qubit_coordinates = {tuple(coord): qubit for qubit, coord in self.s_qubits.items()}
        edges = KDTree(list(qubit_coordinates.keys())).query_pairs(
            Chadoq2.rydberg_blockade_radius(1) * (1 + epsilon),
        )

        return list(edges)

    @classmethod
    def triangle(cls, rows: int, spacing: int):
        qubits = {}
        qubit_index = 0

        horizontal_spacing = spacing * sqrt(3) / 2

        for i in range(rows):
            for j in range(i + 1):
                x = j * horizontal_spacing
                y = i * spacing

                qubits[qubit_index] = (x, y)
                qubit_index += 1

        print(qubits)

        return cls(qubits)


def generate_multiple_configurations(folder: str, max_qubits: int = 14) -> None:
    """Generates multiple configurations of registers.

    Args:
        folder (str): Where to store the configurations
        max_qubits (int, optional): Maximal number of qubits of the configurations. Defaults to 14.
    """

    # Ugly hack to make it work quickly

    for spacing in range(6, 11):
        for i in range(2, max_qubits):
            for j in range(1, max_qubits // i):
                register = GraphRegister.rectangle(rows=i, columns=j, spacing=spacing)
                nb_qubits = str(register.size)
                register.to_json_file(
                    os.path.join(folder, f"{nb_qubits}_rectangle_{i}_{j}_{spacing}.json"),
                )

                register = GraphRegister.triangular_lattice(
                    rows=i,
                    atoms_per_row=j,
                    spacing=spacing,
                )
                nb_qubits = str(register.size)
                register.to_json_file(
                    os.path.join(folder, f"{nb_qubits}_latice_{i}_{j}_{spacing}.json"),
                )

        for l in range(1, int(sqrt(max_qubits))):
            register = GraphRegister.hexagon(layers=l, spacing=spacing)
            nb_qubits = str(register.size)
            register.to_json_file(os.path.join(folder, f"{nb_qubits}_hexagon_{l}_{spacing}.json"))

            register = GraphRegister.triangle(rows=l, spacing=spacing)
            nb_qubits = str(register.size)
            register.to_json_file(os.path.join(folder, f"{nb_qubits}_triangle_{l}_{spacing}.json"))
