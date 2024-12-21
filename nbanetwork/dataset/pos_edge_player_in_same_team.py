# If same team, return positive edge.

import os
from pathlib import Path

import ipdb
import networkx as nx
import pandas as pd
import typer
from loguru import logger
from tqdm import tqdm

from nbanetwork.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    node_input_dir: Path = PROCESSED_DATA_DIR / "players",
    edge_output_dir: Path = INTERIM_DATA_DIR / "players",
    year_from: int = 1996,
    year_until: int = 2021,
    is_debug: bool = False,
):

    input_node_path = str(node_input_dir) + "/" + f"player_nodes_{year_from}-{year_until}.csv"
    # read player nodes
    df_node = pd.read_csv(input_node_path)

    if is_debug:
        df_node = df_node.sample(100)

    # create graph
    G = nx.Graph()

    # add nodes to the graph
    for _, row in df_node.iterrows():
        G.add_node(row["node_id"], **row.to_dict())

    # add edges between players in the same team
    for season in tqdm(df_node["season"].unique(), desc="Adding edges"):
        season_data = df_node[df_node["season"] == season]
        teams = season_data["team_abbreviation"].unique()
        for team in teams:
            team_players = season_data[season_data["team_abbreviation"] == team]["node_id"].tolist()
            # add edges between players in the same team
            for i in range(len(team_players)):
                for j in range(i + 1, len(team_players)):
                    G.add_edge(team_players[i], team_players[j], sign="positive")

    # save edges with positive and negative signs
    edges = list(G.edges(data=True))
    # map node_id to index
    # node_id_to_index = {node_id: idx for idx, node_id in enumerate(G.nodes)}
    # edges = [(node_id_to_index[edge[0]], node_id_to_index[edge[1]], edge[2]) for edge in edges]
    edges_df = pd.DataFrame(edges, columns=["source", "target", "sign"])
    pos_edges_df = edges_df[edges_df["sign"].map(lambda x: x["sign"]) == "positive"][["source", "target"]]
    if not os.path.exists(edge_output_dir):
        os.makedirs(edge_output_dir)
    pos_edge_output_path = str(edge_output_dir) + "/" + f"player_edges_pos_{year_from}-{year_until}.csv"
    pos_edges_df.to_csv(pos_edge_output_path, index=False)

    logger.success("Nodes and edges saved.")


if __name__ == "__main__":
    app()
