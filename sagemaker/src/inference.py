import json
import os
from pathlib import Path

import boto3
import pandas as pd
import torch
import yaml

from nbanetwork.modeling.gnn_models import MultiGraphConv
from nbanetwork.utils import create_data_with_weight, create_node_ids_features_edge_index_with_weight


def download_from_s3(s3_path, local_path):
    s3 = boto3.client("s3")
    bucket, key = s3_path.replace("s3://", "").split("/", 1)
    s3.download_file(bucket, key, local_path)


def model_fn(model_dir):
    config_path = Path(model_dir) / "config.yaml"
    config = load_config(config_path)

    in_channels = config["model"]["params"]["in_channels"]
    hidden_channels = config["model"]["params"]["hidden_channels"]
    dropout = config["model"]["params"]["dropout"]

    model = MultiGraphConv(in_channels, hidden_channels, dropout)
    model_state_path = Path(model_dir) / "model.pth"
    model.load_state_dict(torch.load(model_state_path, map_location=torch.device("cpu")))
    model.eval()
    return model


def input_fn(request_body, request_content_type):
    if request_content_type == "application/json":
        return json.loads(request_body)
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")


def predict_fn(input_data, model):
    ssm_parameters = get_ssm_parameter("nba-players-relation-params")
    bucket_name = json.loads(ssm_parameters)["bucket_name"]
    # S3 パスを指定
    nodes_s3 = f"s3://{bucket_name}/data/node_edges/player_nodes.csv"
    pos_edges_s3 = f"s3://{bucket_name}/data/node_edges/assist_edges_pos.csv"
    neg_edges_s3 = f"s3://{bucket_name}/data/node_edges/assist_edges_neg.csv"

    # ローカルパス
    local_data_dir = "/opt/ml/input/data/node_edges/"
    os.makedirs(local_data_dir, exist_ok=True)
    download_from_s3(nodes_s3, os.path.join(local_data_dir, "player_nodes.csv"))
    download_from_s3(pos_edges_s3, os.path.join(local_data_dir, "assist_edges_pos.csv"))
    download_from_s3(neg_edges_s3, os.path.join(local_data_dir, "assist_edges_neg.csv"))

    nodes_path = Path(local_data_dir) / "player_nodes.csv"
    pos_edge_path = Path(local_data_dir) / "assist_edges_pos.csv"
    neg_edge_path = Path(local_data_dir) / "assist_edges_neg.csv"

    node_ids, features, pos_edge_index, neg_edge_index, pos_edge_weight, neg_edge_weight = (
        create_node_ids_features_edge_index_with_weight(nodes_path, pos_edge_path, neg_edge_path, is_train=False)
    )

    edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)[:2, :]
    edge_weights = torch.cat([pos_edge_weight, neg_edge_weight], dim=0)
    edge_weights = edge_weights / edge_weights.max()
    data = create_data_with_weight(features, edge_index, edge_weights)

    with torch.no_grad():
        z = model(data.x, data.edge_index, data.edge_weight)

        # Load node dataframe
    node_df = pd.read_csv(nodes_path)

    # Define function to predict chemistry between two players
    def predict_chemistry(player1, player2, model, data, node_ids, node_df, z):
        """
        Predicts the chemistry score between two players.

        Args:
            player1 (str): ID or name of the first player.
            player2 (str): ID or name of the second player.
            model (torch.nn.Module): Trained GNN model.
            data (Data): Input data for the model.
            node_ids (dict): Mapping from player IDs to node indices.
            node_df (pd.DataFrame): DataFrame containing player information.
            z (torch.Tensor): Node embeddings.

        Returns:
            float: Chemistry score between player1 and player2.
        """
        # Get node indices for the players
        idx1 = node_ids.get(player1)
        idx2 = node_ids.get(player2)

        if idx1 is None or idx2 is None:
            return 0.0  # Return 0.0 if any player is not found

        # Prepare source and destination tensors
        src = torch.tensor([idx1], dtype=torch.long)
        dst = torch.tensor([idx2], dtype=torch.long)

        # Calculate edge scores using the model's score method
        scores = model.score(z, src, dst)

        # Apply sigmoid to get probability
        chemistry = torch.sigmoid(scores).item()

        return chemistry

    # Predict chemistry for each pair
    players_relation = []
    for i in range(len(players)):
        for j in range(i + 1, len(players)):
            player1 = players[i]
            player2 = players[j]
            chemistry = predict_chemistry(player1, player2, model, data, node_ids, node_df, z)
            players_relation.append({"player1": player1, "player2": player2, "chemistry_score": chemistry})

    return players_relation


def output_fn(prediction, content_type):
    if content_type == "application/json":
        return json.dumps(prediction)
    else:
        raise ValueError(f"Unsupported content type: {content_type}")


def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def get_ssm_parameter(parameter_name, with_decryption=True):
    # SSM clientを作成
    ssm = boto3.client("ssm")

    # パラメータを取得
    response = ssm.get_parameter(Name=parameter_name, WithDecryption=with_decryption)

    # パラメータの値を返す
    return response["Parameter"]["Value"]
