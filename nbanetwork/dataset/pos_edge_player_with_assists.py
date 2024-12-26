# If assist data is high quality, we can use it to create positive edges between players.

import os
from collections import defaultdict
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
    input_path: Path = INTERIM_DATA_DIR / "players" / "assist_relation.csv",
    node_input_dir: Path = PROCESSED_DATA_DIR / "players",
    edge_output_dir: Path = INTERIM_DATA_DIR / "players",
    year_from: int = 1996,
    year_until: int = 2022,
    is_debug: bool = False,
):
    input_node_path = str(node_input_dir) + "/" + f"player_nodes_{year_from}-{year_until}.csv"
    # read player nodes
    df_node = pd.read_csv(input_node_path)

    if is_debug:
        df_node = df_node.sample(1000)

    # create graph
    G = nx.Graph()

    # add nodes to the graph
    for _, row in df_node.iterrows():
        G.add_node(row["node_id"], **row.to_dict())

    # add edges if the number of assists is greater than the threshold
    assist_data = pd.read_csv(input_path)
    assist_data["season"] = assist_data["season"].fillna(method="ffill")
    assist_data = assist_data[assist_data["season"].map(lambda x: int(x[:4])) >= year_from]
    assist_data = assist_data[assist_data["season"].map(lambda x: int(x[:4])) < year_until]
    assist_data = assist_data[assist_data["assister"].notnull()]
    assist_data = assist_data[assist_data["scorer"].notnull()]
    assist_data = assist_data[assist_data["assister"] != assist_data["scorer"]]

    if is_debug:
        assist_data = assist_data.sample(1000)
    # save edges with positive and negative signs
    for _, group in tqdm(assist_data.groupby("season"), desc="Adding edges"):
        for _, row in group.iterrows():
            scorer = df_node[df_node["player_name"] == row["scorer"]]["node_id"].values
            assister = df_node[df_node["player_name"] == row["assister"]]["node_id"].values
            if len(scorer) == 0 or len(assister) == 0:
                continue
            scorer = scorer[0]
            assister = assister[0]
            if scorer == assister:
                continue
            if G.has_edge(assister, scorer):
                G[assister][scorer]["weight"] += 1
            else:
                G.add_edge(assister, scorer, weight=1)

    # save edge list
    output_path = edge_output_dir / f"assist_edges_pos_{year_from}-{year_until}.csv"
    with open(output_path, "w") as f:
        f.write("source,target,weight\n")
        for u, v, d in G.edges(data=True):
            f.write(f"{u},{v},{d['weight']}\n")
