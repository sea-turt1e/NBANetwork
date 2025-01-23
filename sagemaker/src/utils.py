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
    # if is_train:
    #     node_df = node_df.sample(frac=1).reset_index(drop=True)
    #     pos_edge_df = pos_edge_df.sample(frac=1).reset_index(drop=True)
    #     neg_edge_df = neg_edge_df.sample(frac=1).reset_index(drop=True)

    # map node ids
    node_ids = {name: idx for idx, name in enumerate(node_df["node_id"])}

    # create labels (has edge or not)
    pos_edge = [
        [node_ids.get(row["source"]), node_ids.get(row["target"])]
        for idx, row in pos_edge_df.iterrows()
        if node_ids.get(row["source"]) is not None and node_ids.get(row["target"]) is not None
    ]
    neg_edge = [
        [node_ids.get(row["source"]), node_ids.get(row["target"])]
        for idx, row in neg_edge_df.iterrows()
        if node_ids.get(row["source"]) is not None and node_ids.get(row["target"]) is not None
    ]
    # create edge index
    pos_edge_index = torch.tensor(pos_edge, dtype=torch.long).t().contiguous()
    neg_edge_index = torch.tensor(neg_edge, dtype=torch.long).t().contiguous()

    # create features
    drop_node_df_columns = ["node_id", "player_name"]
    node_df = node_df.drop(drop_node_df_columns, axis=1)
    drop_columns = ["team_abbreviation", "college", "country", "season", "draft_year", "draft_round"]
    # for col in drop_columns:
    #     node_df[col] = node_df[col].astype("category").cat.codes

    numerical_columns = [col for col in node_df.columns if col not in drop_columns]
    features = node_df[numerical_columns]
    # features = node_df[categorical_columns + numerical_columns]
    features = torch.tensor(features.values, dtype=torch.float)
    return node_ids, features, pos_edge_index, neg_edge_index


def create_node_ids_features_edge_index_with_weight(
    nodes_path: str, pos_edge_path: str, neg_edge_path: str, is_train: bool = True
):
    # read attributes of nodes and edges
    node_df = pd.read_csv(nodes_path)
    pos_edge_df = pd.read_csv(pos_edge_path)
    neg_edge_df = pd.read_csv(neg_edge_path)
    # sfuhhle if is_train=True.
    # if is_train:
    #     node_df = node_df.sample(frac=1).reset_index(drop=True)
    #     pos_edge_df = pos_edge_df.sample(frac=1).reset_index(drop=True)
    #     neg_edge_df = neg_edge_df.sample(frac=1).reset_index(drop=True)

    # map node ids
    node_ids = {name: idx for idx, name in enumerate(node_df["node_id"])}

    # create labels (has edge or not)
    pos_edge = [
        [node_ids.get(row["source"]), node_ids.get(row["target"]), row["weight"]]
        for idx, row in pos_edge_df.iterrows()
        if node_ids.get(row["source"]) is not None and node_ids.get(row["target"]) is not None
    ]
    neg_edge = [
        [node_ids.get(row["source"]), node_ids.get(row["target"]), row["weight"]]
        for idx, row in neg_edge_df.iterrows()
        if node_ids.get(row["source"]) is not None and node_ids.get(row["target"]) is not None
    ]
    # create edge index
    pos_edge_index = torch.tensor(pos_edge, dtype=torch.long).t().contiguous()
    pos_weights = pos_edge_index[2]
    pos_edge_index = pos_edge_index[:2, :]
    neg_edge_index = torch.tensor(neg_edge, dtype=torch.long).t().contiguous()
    neg_weights = neg_edge_index[2]
    neg_edge_index = neg_edge_index[:2, :]
    # create features
    drop_node_df_columns = ["node_id", "player_name"]
    node_df = node_df.drop(drop_node_df_columns, axis=1)
    drop_columns = ["team_abbreviation", "college", "country", "season", "draft_year", "draft_round"]
    # for col in drop_columns:
    #     node_df[col] = node_df[col].astype("category").cat.codes

    numerical_columns = [col for col in node_df.columns if col not in drop_columns]
    features = node_df[numerical_columns]
    # features = node_df[categorical_columns + numerical_columns]
    features = torch.tensor(features.values, dtype=torch.float)
    return node_ids, features, pos_edge_index, neg_edge_index, pos_weights, neg_weights


def create_data(features: torch.Tensor, edge_index: torch.Tensor):
    return Data(x=features, edge_index=edge_index)


def create_data_with_weight(features: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor):
    return Data(x=features, edge_index=edge_index, edge_weight=edge_weight)
