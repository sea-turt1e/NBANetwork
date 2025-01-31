import os
from pathlib import Path

import ipdb
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import torch
import typer
from loguru import logger

from nbanetwork.config import MODELS_DIR, PROCESSED_DATA_DIR
from nbanetwork.modeling.gnn_models import GCN
from nbanetwork.utils import create_data, create_node_ids_features_edge_index

app = typer.Typer()


@app.command()
def main(
    node_edges_date_dir: Path = PROCESSED_DATA_DIR / "players",
    pos_neg_edges_dir: Path = PROCESSED_DATA_DIR / "players",
    model_path: Path = MODELS_DIR / "gnn_model.pth",
    output_plot_dir: Path = PROCESSED_DATA_DIR / "plots",
    year_from: int = 2022,
    year_until: int = 2023,
):

    # create features and edge index
    nodes_path = node_edges_date_dir / f"player_nodes_{year_from}-{year_until}.csv"
    pos_edge_path = pos_neg_edges_dir / f"player_edges_pos_{year_from}-{year_until}.csv"
    neg_edge_path = pos_neg_edges_dir / f"player_edges_neg_{year_from}-{year_until}.csv"

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

    # Define function to predict chemistry between two players
    def predict_chemistry(player1, player2, model, data, node_ids, node_df, z):
        """
        Predicts the chemistry score between two players.

        Args:
            player1 (str): ID or name of the first player.
            player2 (str): ID or name of the second player.
            model (GCN): Trained GNN model.
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

        # Get team information for the players
        player1_team = node_df.loc[node_df["node_id"] == player1, "team_abbreviation"].values
        player2_team = node_df.loc[node_df["node_id"] == player2, "team_abbreviation"].values

        if len(player1_team) > 0 and len(player2_team) > 0:
            if player1_team[0] == player2_team[0]:
                return 1.0  # Return 1.0 if players are on the same team

        # Calculate edge scores using the model's score method
        src = torch.tensor([idx1], dtype=torch.long)
        dst = torch.tensor([idx2], dtype=torch.long)
        scores = model.score(z, src, dst)

        # Apply sigmoid to get probability
        chemistry = torch.sigmoid(scores).item()

        return chemistry

    # pick up common players from the prediction file
    pickup_players_name = [
        "Stephen Curry_2009_7",
        "James Harden_2009_3",
        "Jimmy Butler_2011_30",
        "Nikola Jokic_2014_41",
        "Rui Hachimura_2019_9",
        "Kawhi Leonard_2011_15",
    ]

    players_relation = []
    # predict chemistry for each pair of common players
    for i in range(len(pickup_players_name)):
        for j in range(i + 1, len(pickup_players_name)):
            player1 = pickup_players_name[i]
            player2 = pickup_players_name[j]
            chemistry = predict_chemistry(player1, player2, model, data, node_ids, node_df, z)
            players_relation.append([player1, player2, chemistry])

    # visualize chemistry network
    logger.info("Visualizing chemistry network...")
    G = nx.Graph()
    for row in players_relation:
        G.add_edge(row[0], row[1], weight=round(float(row[2]), 3))

    pos = nx.spring_layout(G, k=0.15, iterations=20)  # Adjust layout parameters for better spacing
    edge_labels = nx.get_edge_attributes(G, "weight")

    plt.figure(figsize=(15, 10))  # Increase figure size for better readability

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=150, node_color="skyblue", alpha=0.7)

    # Draw labels with smaller font size and slight offset
    for node, (x, y) in pos.items():
        plt.text(x, y + 0.02, node, fontsize=8, ha="center", va="bottom")

    # nx.draw_networkx_edges(G, pos, alpha=0.5)

    # If weight is higher than 0.95, draw edge label with red color
    red_edges = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] > 0.95 and d["weight"] <= 1.0]
    black_edges = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] > 0.30 and d["weight"] <= 0.95]
    blue_edges = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] <= 0.30]
    # If 1,0, don't draw edge label
    edge_labels = {(u, v): f"{d['weight']:.2f}" for (u, v, d) in G.edges(data=True) if d["weight"] != 1.0}

    # Draw edges
    nx.draw_networkx_edges(G, pos, edgelist=red_edges, edge_color="r", width=2)
    nx.draw_networkx_edges(G, pos, edgelist=black_edges, edge_color="black", width=1)
    nx.draw_networkx_edges(G, pos, edgelist=blue_edges, edge_color="blue", width=1)

    # Draw edge labels with smaller font size
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6, label_pos=0.3)

    # Create output directory if it doesn't exist
    if not os.path.exists(output_plot_dir):
        os.makedirs(output_plot_dir)

    plt.axis("off")  # Hide axes for a cleaner look
    plt.tight_layout()  # Adjust subplot params for better layout
    plt.savefig(output_plot_dir / "chemistry_network.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    app()
