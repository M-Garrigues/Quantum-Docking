import os

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data, Dataset

from src.solver.utils.graph_register import GraphRegister


class CustomGraphDataset(Dataset):
    """A dataset object designed to deal with registers data."""

    def __init__(self, folder_path, split_ratio=0.8, transform=None, target_index: int = 0):
        self.target_index = target_index
        self.adjacency_matrices, self.target_values = self.get_adjacency_matrices(folder_path)
        self._indices = None
        self.transform = transform

        (
            self.train_adj_matrices,
            self.test_adj_matrices,
            self.train_targets,
            self.test_targets,
        ) = train_test_split(
            self.adjacency_matrices,
            self.target_values,
            test_size=1 - split_ratio,
            random_state=42,
        )

        # Flag to indicate whether to use train or test split
        self.use_train_split = True

    def len(self):
        """Returns the size of the current dataset."""
        if self.use_train_split:
            return len(self.train_adj_matrices)
        else:
            return len(self.test_adj_matrices)

    def get(self, idx: int) -> Data:
        """Overload dataset get function.
        Returns a pytorch Data object with graph info and target."""
        if self.use_train_split:
            adjacency_matrix = self.train_adj_matrices[idx]
            target_value = self.train_targets[idx]
        else:
            adjacency_matrix = self.test_adj_matrices[idx]
            target_value = self.test_targets[idx]

        adjacency_matrix = self.adjacency_matrices[idx]
        target_value = torch.Tensor([self.target_values[idx]])

        data = self.data_from_matrix(adjacency_matrix, target=target_value)

        return data

    @staticmethod
    def data_from_matrix(adjacency_matrix: np.ndarray, target: torch.Tensor = None) -> Data:
        """Generates a Data object from an adjacency matrix,
        optionally with a target.

        Args:
            adjacency_matrix (np.ndarray): Graph full adjacency matrix.
            target (torch.Tensor, optional):
                The associated target tensor for training purposes. Defaults to None.

        Returns:
            Data: Pytorch geometric data object with graph info.
        """
        adjacency_matrix = torch.Tensor(adjacency_matrix)
        edge_index = torch.nonzero(adjacency_matrix, as_tuple=True)

        # Necessary trick in order to make the edges undirected
        edge_from = torch.concat((edge_index[0], edge_index[1]))
        edge_to = torch.concat((edge_index[1], edge_index[0]))

        edge_index = torch.stack((edge_from, edge_to), dim=0)

        edge_weight = adjacency_matrix.view(-1, 1).float()  # type: ignore
        edge_weight = edge_weight[edge_weight != 0]

        # Same as before for the undirected edges
        edge_weight = torch.concat((edge_weight, edge_weight))

        if not target:
            data = Data(
                edge_index=edge_index,
                edge_weight=edge_weight,
                num_nodes=len(adjacency_matrix),
            )
        else:
            data = Data(
                edge_index=edge_index,
                edge_weight=edge_weight,
                y=target,
                num_nodes=len(adjacency_matrix),
            )
        return data

    def get_adjacency_matrices(self, folder_path):
        """Gets the adjacency matrices for all registers files in a folder."""
        adjacency_matrices = []
        target_values = []

        for filename in os.listdir(folder_path):
            if filename.endswith(".json"):
                file_path = os.path.join(folder_path, filename)
                adjacency_matrix, target_value = data_from_file(file_path, self.target_index)
                adjacency_matrices.append(adjacency_matrix)
                target_values.append(target_value)

        return adjacency_matrices, target_values


def data_from_file(file_path: str, target_index: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Returns the adjacency matrix and a single target value for a single file."""
    register = GraphRegister.from_json(file_path)
    qubits = register.qubits
    adjacency_matrix = create_adjacency_matrix(list(qubits.values()))
    targets = np.array(list(register.metadata["params"].values()))
    return adjacency_matrix, targets[target_index]


def calculate_distance(coord1, coord2):
    return np.sqrt((coord2[0] - coord1[0]) ** 2 + (coord2[1] - coord1[1]) ** 2)


def van_der_waals_interaction(distance, epsilon, sigma):
    return 4 * epsilon * ((sigma / distance) ** 12 - (sigma / distance) ** 6)


def create_adjacency_matrix(atoms_coords, epsilon=1, sigma=1):
    """Generates a full adjacency matrix for a given set of coordinates.
    The weight of each edge is the inverse of the square of the distance."""
    num_atoms = len(atoms_coords)
    adjacency_matrix = np.zeros((num_atoms, num_atoms))

    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            distance = calculate_distance(atoms_coords[i], atoms_coords[j])
            interaction_strength = distance  # van_der_waals_interaction(distance, epsilon, sigma)

            adjacency_matrix[i, j] = 1 / (interaction_strength**2)
            adjacency_matrix[j, i] = 1 / (interaction_strength**2)

    return adjacency_matrix
