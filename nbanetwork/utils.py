import ast

import ipdb
import pandas as pd
import torch
from loguru import logger
from torch_geometric.data import Data
from tqdm import tqdm


def create_node_ids_features_edge_index(nodes_path: str, pos_edge_path: str, neg_edge_path: str, is_train: bool = True):
    # read attributes of nodes and edges
    node_df = pd.read_csv(nodes_path)
    pos_edge_df = pd.read_csv(pos_edge_path)
    neg_edge_df = pd.read_csv(neg_edge_path)
    # sfuhhle if is_train=True.
    if is_train:
        node_df = node_df.sample(frac=1).reset_index(drop=True)
        pos_edge_df = pos_edge_df.sample(frac=1).reset_index(drop=True)
        neg_edge_df = neg_edge_df.sample(frac=1).reset_index(drop=True)

    # map node ids
    node_ids = {name: idx for idx, name in enumerate(node_df["node_id"])}
    pos_source = pos_edge_df["source"].map(node_ids).tolist()
    pos_target = pos_edge_df["target"].map(node_ids).tolist()
    neg_source = neg_edge_df["source"].map(node_ids).tolist()
    neg_target = neg_edge_df["target"].map(node_ids).tolist()

    # create edge index
    pos_edge = list(zip(pos_source, pos_target))
    neg_edge = list(zip(neg_source, neg_target))

    # create labels (has edge or not)
    pos_edge_index = torch.tensor(pos_edge, dtype=torch.long).t().contiguous()
    neg_edge_index = torch.tensor(neg_edge, dtype=torch.long).t().contiguous()

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

    return node_ids, features, pos_edge_index, neg_edge_index


def create_data(features: torch.Tensor, edge_index: torch.Tensor):
    return Data(x=features, edge_index=edge_index)
