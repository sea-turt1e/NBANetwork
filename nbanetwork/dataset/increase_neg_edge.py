import os
import random
from pathlib import Path

import ipdb
import pandas as pd
import typer
from loguru import logger
from tqdm import tqdm

app = typer.Typer()


@app.command()
def main(
    nodes_path: str,
    pos_edge_path: str,
    neg_edge_path: str,
    output_dir: str,
    edge_ratio: float = 1.0,  # ratio of negative samples to positive samples
    is_debug: bool = False,
):
    if edge_ratio < 1.0:
        logger.error("edge_ratio must be greater than or equal to 1.0.")
        return
    elif edge_ratio > 10.0:
        logger.error("edge_ratio must be less than or equal to 10.0.")
        return
    node_df = pd.read_csv(nodes_path)
    num_nodes = len(node_df)
    pos_edges = pd.read_csv(pos_edge_path)
    if not os.path.exists(neg_edge_path):
        with open(neg_edge_path, "w") as f:
            f.write("source,target\n")
    neg_edges = pd.read_csv(neg_edge_path)
    # increase negative samples
    logger.info("Increasing negative samples...")
    if is_debug:
        pos_edges = pos_edges[:100]

    pos_edges_list = pos_edges[["source", "target"]].values.tolist()
    neg_edges_list = neg_edges[["source", "target"]].values.tolist()

    num_neg_edges_increase = int(len(pos_edges) * edge_ratio - len(neg_edges))
    print(f"num_edges_increase: {num_neg_edges_increase}")
    num_while = 0
    num_neg_edges = 0
    # save positive and negative samples
    output_pos_edge_path = output_dir + "/" + os.path.basename(pos_edge_path)
    output_neg_edge_path = output_dir + "/" + os.path.basename(neg_edge_path)
    # convert list to set for faster search
    pos_edges_set = set(tuple(edge) for edge in pos_edges_list)
    neg_edges_set = set(tuple(edge) for edge in neg_edges_list)
    with open(output_neg_edge_path, "w") as f:
        f.write("source,target\n")
        with tqdm() as pbar:
            while num_neg_edges < num_neg_edges_increase:
                i, j = random.randint(0, num_nodes - 1), random.randint(0, num_nodes - 1)
                # if edge does not exist
                if i == j:
                    player1 = node_df.iloc[i]["node_id"]
                    player2 = node_df.iloc[j]["node_id"]
                    if (player1, player2) not in (pos_edges_set | neg_edges_set):
                        neg_edges_list.append([player1, player2])
                        f.write(f"{player1},{player2}\n")
                        num_neg_edges += 1
                        pbar.update(1)
                    num_while += 1
                if num_while > num_neg_edges_increase * 10:
                    logger.error("too many while loops.")
                    break
        logger.info("Negative samples increased.")
    with open(output_pos_edge_path, "w") as f:
        f.write("source,target\n")
        for edge in pos_edges_list:
            f.write(f"{edge[0]},{edge[1]}\n")
    logger.success("Positive and negative samples created.")
