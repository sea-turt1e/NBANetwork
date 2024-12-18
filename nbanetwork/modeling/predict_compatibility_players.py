import csv
from pathlib import Path

import ipdb
import typer
from loguru import logger
from tqdm import tqdm

from nbanetwork.config import MODELS_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR
from nbanetwork.modeling.gnn_models import GCN

app = typer.Typer()


@app.command()
def main(
    node_edges_date_dir: Path = PROCESSED_DATA_DIR / "players",
    pos_neg_edges_dir: Path = PROCESSED_DATA_DIR / "players",
    year_from: int = 2021,
    year_until: int = 2023,
    model_path: Path = MODELS_DIR / "gnn_model.pth",
    predictions_dir: Path = PROCESSED_DATA_DIR / "predictions",
    is_debug: bool = False,
):
    import os

    import pandas as pd
    import torch

    from nbanetwork.utils import create_data, create_node_ids_features_edge_index

    # create features and edge index
    nodes_path = node_edges_date_dir / f"player_nodes_{year_from}-{year_until}.csv"
    pos_edge_path = pos_neg_edges_dir / f"players_pos_edge_{year_from}-{year_until}.txt"
    neg_edge_path = pos_neg_edges_dir / f"players_neg_edge_{year_from}-{year_until}.txt"

    node_ids, features, pos_edge_index, neg_edge_index = create_node_ids_features_edge_index(
        nodes_path, pos_edge_path, neg_edge_path, is_train=False
    )

    # create data
    edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)

    # create data
    data = create_data(features, edge_index)

    model = GCN(in_channels=features.shape[1], hidden_channels=64)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    # get embeddings of nodes
    with torch.no_grad():
        z = model(data.x, data.edge_index)

    node_df = pd.read_csv(nodes_path)

    # define function to predict compatibility
    def predict_compatibility(player1, player2):
        # if player1 and player2 are in same team
        player1_team = node_df[node_df["node_id"] == player1]["team_abbreviation"].values[0]
        player2_team = node_df[node_df["node_id"] == player2]["team_abbreviation"].values[0]
        if player1_team == player2_team:
            return 99
        idx1 = node_ids.get(player1)
        idx2 = node_ids.get(player2)
        if idx1 is None or idx2 is None:
            return "cannot find player"
        emb1 = z[idx1]
        emb2 = z[idx2]
        score = torch.sigmoid((emb1 * emb2).sum()).item()
        return score

    # Example: check compatibility
    e_p1 = "Stephen Curry_2009_7"
    e_p2 = "Rui Hachimura_2019_9"
    example_comptibility = predict_compatibility(e_p1, e_p2)
    print(f'Example: Compatibility between "{e_p1}" and "{e_p2}" is {example_comptibility}')

    # save predictions
    predictions_path = predictions_dir / f"compatibility_{year_from}-{year_until}.csv"
    prediction_common_player_path = predictions_dir / f"compatibility_common_player_{year_from}-{year_until}.csv"
    if not os.path.exists(predictions_dir):
        os.makedirs(predictions_dir)
    with (
        open(predictions_path, "w") as fw_all,
        open(prediction_common_player_path, "w") as fw_common,
        open(RAW_DATA_DIR / "nbadatabase" / "csv" / "common_player_info.csv", "r") as fr_common,
    ):
        fw_all.write("player1,player2,score\n")
        fw_common.write("player1,player2,score\n")
        reader = csv.reader(fr_common, delimiter=",")
        next(reader)
        common_players_name = [row[3] for row in reader]
        if is_debug:
            node_ids = {node[0]: node[1] for node in list(node_ids.items())[-10:]}
        for player1 in tqdm(node_ids):
            for player2 in node_ids:
                if player1 != player2:
                    score = predict_compatibility(player1, player2)
                    fw_all.write(f"{player1},{player2},{score}\n")
                    if {player1[:-8], player2[:-8]} & set(common_players_name):
                        fw_common.write(f"{player1},{player2},{score}\n")
    logger.success("Predictions saved.")


if __name__ == "__main__":
    app()
