import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, SAGEConv


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


class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, dropout=0.5):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels * 2)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels * 2)
        self.conv3 = SAGEConv(hidden_channels * 2, hidden_channels * 2)
        self.bn3 = torch.nn.BatchNorm1d(hidden_channels * 2)
        self.dropout = dropout
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels * 4, hidden_channels * 2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels * 2, 1),
        )

    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.bn1(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.bn2(x)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.bn3(x)
        return x

    def score(self, z, src, dst):
        edge_emb = torch.cat([z[src], z[dst]], dim=1)
        scores = self.fc(edge_emb).squeeze()
        return scores


class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels=128, heads=4, dropout=0.5):
        super(GAT, self).__init__()
        # conv1: 出力次元 = hidden_channels × heads
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels * heads)

        # conv2: 入力 = hidden_channels × heads, 出力 次元 = out_channels × heads
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=heads)
        self.bn2 = torch.nn.BatchNorm1d(out_channels * heads)

        # conv3: 入力 = out_channels × heads, 出力 次元 = out_channels × heads
        self.conv3 = GATConv(out_channels * heads, out_channels, heads=heads)
        self.bn3 = torch.nn.BatchNorm1d(out_channels * heads)
        self.fc = torch.nn.Sequential(torch.nn.Linear(1024, 128), torch.nn.ReLU(), torch.nn.Linear(128, 1))

        self.dropout = dropout

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
