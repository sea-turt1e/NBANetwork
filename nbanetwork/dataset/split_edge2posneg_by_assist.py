import os
import random

import ipdb
import pandas as pd
import typer
from loguru import logger
from tqdm import tqdm

app = typer.Typer()


@app.command()
def main(
    edge_path: str,
    output_dir: str,
    assist_threshold: int = 1,  # threshold for assist of positive samples
    is_debug: bool = False,
):
    edges = pd.read_csv(edge_path)
    # increase negative samples
    logger.info("Increasing negative samples...")
    if is_debug:
        edges = edges[:100]

    pos_edges_list = [
        edge for edge in edges[["source", "target", "weight"]].values.tolist() if edge[2] > assist_threshold
    ]
    neg_edges_list = [
        edge for edge in edges[["source", "target", "weight"]].values.tolist() if edge[2] <= assist_threshold
    ]

    logger.info("positive_edges:", len(pos_edges_list))
    logger.info("negative_edges:", len(neg_edges_list))
    # save positive and negative samples
    output_pos_edge_path = output_dir + "/same_team_pos_" + os.path.basename(edge_path)
    output_neg_edge_path = output_dir + "/same_team_neg_" + os.path.basename(edge_path)

    with open(output_pos_edge_path, "w") as f:
        f.write("source,target,weight\n")
        for edge in pos_edges_list:
            f.write(f"{edge[0]},{edge[1]},{edge[2]}\n")
    logger.info(f"Saved positive samples to {output_pos_edge_path}")

    with open(output_neg_edge_path, "w") as f:
        f.write("source,target,weight\n")
        for edge in neg_edges_list:
            f.write(f"{edge[0]},{edge[1]},{edge[2]}\n")
    logger.info(f"Saved negative samples to {output_neg_edge_path}")
