from pathlib import Path

import ipdb
import typer
from loguru import logger
from tqdm import tqdm

from nbanetwork.config import INTERIM_DATA_DIR, MODELS_DIR, PROCESSED_DATA_DIR
from nbanetwork.modeling.gnn_models import GCN

app = typer.Typer()


@app.command()
def main(
    node_edges_date_dir: Path = INTERIM_DATA_DIR / "players",
    year_from: int = 2021,
    year_until: int = 2023,
    model_path: Path = MODELS_DIR / "gnn_model.pth",
    predictions_dir: Path = PROCESSED_DATA_DIR / "predictions",
):
    import os

    import pandas as pd
    import torch

    from nbanetwork.utils import create_data, create_node_ids_features_edge_index

    # create features and edge index
    nodes_path = node_edges_date_dir / f"player_nodes_{year_from}-{year_until}_normalized.csv"
    edge_path = node_edges_date_dir / f"player_edges_{year_from}-{year_until}.csv"
    node_ids, features, edge_index = create_node_ids_features_edge_index(nodes_path, edge_path)

    # create data
    data = create_data(features, edge_index)

    model = GCN(in_channels=features.shape[1], hidden_channels=64)
    model.load_state_dict(torch.load(model_path))
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

    # Example: check compability
    # print(predict_compatibility("Stephen Curry_2022-23", "Kevin Durant_2022-23"))

    # save predictions
    predictions_path = predictions_dir / f"compatibility_{year_from}-{year_until}.csv"
    if not os.path.exists(predictions_dir):
        os.makedirs(predictions_dir)
    with open(predictions_path, "w") as f:
        f.write("player1,player2,score\n")
        for player1 in tqdm(node_ids):
            for player2 in node_ids:
                if player1 != player2:
                    score = predict_compatibility(player1, player2)
                    f.write(f"{player1},{player2},{score}\n")


if __name__ == "__main__":
    app()
