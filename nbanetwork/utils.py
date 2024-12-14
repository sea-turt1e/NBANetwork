import ipdb
import pandas as pd
import torch
from torch_geometric.data import Data


def create_node_ids_features_edge_index(nodes_path: str, edge_path: str, is_train: bool = True):
    # read attributes of nodes and edges
    node_df = pd.read_csv(nodes_path)
    edge_df = pd.read_csv(edge_path)
    # sfuhhle if is_train=True.
    if is_train:
        node_df = node_df.sample(frac=1).reset_index(drop=True)
        edge_df = edge_df.sample(frac=1).reset_index(drop=True)

    # map node ids
    node_ids = {name: idx for idx, name in enumerate(node_df["node_id"])}
    source = edge_df["source"].map(node_ids).tolist()
    target = edge_df["target"].map(node_ids).tolist()

    # create features
    drop_node_df_columns = ["node_id", "player_name"]
    node_df = node_df.drop(drop_node_df_columns, axis=1)
    categorical_columns = ["team_abbreviation", "college", "country", "season"]
    # for col in categorical_columns:
    #     node_df[col] = node_df[col].astype("category").cat.codes

    numerical_columns = [col for col in node_df.columns if col not in categorical_columns]
    features = node_df[numerical_columns]
    # features = node_df[categorical_columns + numerical_columns]
    features = torch.tensor(features.values, dtype=torch.float)

    # create labels (has edge or not)
    edge_index = torch.tensor([source, target], dtype=torch.long)

    return node_ids, features, edge_index


def create_data(features: torch.Tensor, edge_index: torch.Tensor):
    return Data(x=features, edge_index=edge_index)
