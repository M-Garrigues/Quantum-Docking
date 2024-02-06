import os
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv, global_mean_pool

from src.solver.ml.dataset import CustomGraphDataset


class GraphRegressionModel(nn.Module):
    """Graph Convolution Netowrk using 5 convolutions layers
    to generate graph encodings, used in a single target regression task."""

    def __init__(self, hidden_channels):
        super().__init__()
        torch.manual_seed(12345)

        self.conv1 = GCNConv(-1, hidden_channels, normalize=False, add_self_loops=True)
        self.conv2 = GCNConv(hidden_channels, hidden_channels, normalize=False, add_self_loops=True)
        self.conv3 = GCNConv(hidden_channels, hidden_channels, normalize=False, add_self_loops=True)
        self.conv4 = GCNConv(hidden_channels, hidden_channels, normalize=False, add_self_loops=True)
        self.conv5 = GCNConv(hidden_channels, hidden_channels, normalize=False, add_self_loops=True)

        self.lin = Linear(hidden_channels, 1)

    def forward(self, data):
        x, edge_index, edge_weight = (
            torch.ones(data.num_nodes, 1),
            data.edge_index,
            data.edge_weight,
        )

        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = x.relu()
        x = F.dropout(x, p=0.1, training=self.training)

        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        x = x.relu()
        x = F.dropout(x, p=0.1, training=self.training)

        x = self.conv3(x, edge_index, edge_weight=edge_weight)
        x = x.relu()
        x = F.dropout(x, p=0.1, training=self.training)

        x = self.conv4(x, edge_index, edge_weight=edge_weight)
        x = x.relu()
        x = F.dropout(x, p=0.1, training=self.training)

        x = self.conv5(x, edge_index, edge_weight=edge_weight)
        x = x.relu()
        x = F.dropout(x, p=0.1, training=self.training)

        x = global_mean_pool(x, data.batch)  # [batch_size, hidden_channels]

        x = self.lin(x)
        return x


@dataclass
class ParamsPrediction:
    rise_time: float | None = None
    fall_time: float | None = None
    omega: float | None = None
    init_detuning: float | None = None
    final_detuning: float | None = None

    @property
    def dict(self) -> dict:
        return asdict(self)


class ParametersModel:
    """Model aggregating the five different parameters models."""

    def __init__(self, models_folder_path: str) -> None:
        self.rise_time_model = torch.load(
            os.path.join(models_folder_path, "rise_time.model"),
        ).eval()
        self.fall_time_model = torch.load(
            os.path.join(models_folder_path, "fall_time.model"),
        ).eval()
        self.omega_model = torch.load(os.path.join(models_folder_path, "omega.model")).eval()
        self.init_detuning_model = torch.load(
            os.path.join(models_folder_path, "init_detuning.model"),
        ).eval()
        self.final_detuning_model = torch.load(
            os.path.join(models_folder_path, "final_detuning.model"),
        ).eval()

    def predict(self, adjacency_matrix: np.ndarray) -> ParamsPrediction:
        """Predicts the parmaters for a given adjacency matrix."""
        prediction = ParamsPrediction()

        data = CustomGraphDataset.data_from_matrix(adjacency_matrix)

        with torch.no_grad():
            prediction.rise_time = self.rise_time_model(data).item()
            prediction.fall_time = self.fall_time_model(data).item()
            prediction.omega = self.omega_model(data).item()
            prediction.init_detuning = self.init_detuning_model(data).item()
            prediction.final_detuning = self.final_detuning_model(data).item()

        return prediction
