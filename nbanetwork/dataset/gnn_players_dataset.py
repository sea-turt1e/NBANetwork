import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data


class GNNPlayersDataset(Dataset):
    def __init__(self, edges, labels, weights, x):
        self.edges = edges
        self.labels = labels
        self.weights = weights
        self.x = x

    def __len__(self):
        return len(self.edges)

    def __getitem__(self, idx):
        edge = self.edges[idx]
        label = self.labels[idx]
        weight = self.weights[idx]
        return Data(x=self.x, edge_index=edge, edge_weight=weight, y=label)
