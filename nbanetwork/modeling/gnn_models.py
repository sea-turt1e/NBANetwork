import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, dropout=0.5):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels * 2)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels * 2)
        self.conv3 = GCNConv(hidden_channels * 2, hidden_channels * 2)
        self.bn3 = torch.nn.BatchNorm1d(hidden_channels * 2)
        self.dropout = dropout
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels * 4, hidden_channels * 2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels * 2, 1),
        )

    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.bn1(x)
        x = self.conv2(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.bn2(x)
        x = self.conv3(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.bn3(x)
        return x

    def score(self, z, src, dst):
        edge_emb = torch.cat([z[src], z[dst]], dim=1)
        scores = self.fc(edge_emb).squeeze()
        return scores


class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, heads=8):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=0.6)
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=1, concat=False, dropout=0.6)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return x
