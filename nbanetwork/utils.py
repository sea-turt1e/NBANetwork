import pandas as pd
import torch
from torch_geometric.data import Data


def create_node_ids_features_edge_index(nodes_path: str, edge_path: str):
    # read attributes of nodes and edges
    node_df = pd.read_csv(nodes_path)
    edge_df = pd.read_csv(edge_path)

    # map node ids
    node_ids = {name: idx for idx, name in enumerate(node_df["node_id"])}
    source = edge_df["source"].map(node_ids).tolist()
    target = edge_df["target"].map(node_ids).tolist()

    # create features
    # use only numerical data
    features = node_df.drop(["node_id", "player_name", "team_abbreviation", "college", "country", "season"], axis=1)
    features = torch.tensor(features.values, dtype=torch.float)

    # create labels (has edge or not)
    edge_index = torch.tensor([source, target], dtype=torch.long)

    return node_ids, features, edge_index


def create_data(features: torch.Tensor, edge_index: torch.Tensor):
    return Data(x=features, edge_index=edge_index)
