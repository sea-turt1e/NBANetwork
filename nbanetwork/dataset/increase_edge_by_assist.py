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
    node_team_map = dict(zip(node_df["node_id"], node_df["team_abbreviation"]))
    num_nodes = len(node_df)
    pos_edges = pd.read_csv(pos_edge_path)
    if not os.path.exists(neg_edge_path):
        with open(neg_edge_path, "w") as f:
            f.write("source,target,weight\n")
    neg_edges = pd.read_csv(neg_edge_path)
    # increase negative samples
    logger.info("Increasing negative samples...")
    if is_debug:
        pos_edges = pos_edges[:100]

    pos_edges_list = pos_edges[["source", "target", "weight"]].values.tolist()
    neg_edges_list = neg_edges[["source", "target", "weight"]].values.tolist()
    # convert list to set for faster search
    pos_edges_set = set(tuple(edge) for edge in pos_edges_list)
    neg_edges_set = set(tuple(edge) for edge in neg_edges_list)

    num_neg_edges_increase = int(len(pos_edges) * edge_ratio - len(neg_edges))
    print(f"num_edges_increase: {num_neg_edges_increase}")
    # save positive and negative samples
    output_pos_edge_path = output_dir + "/" + os.path.basename(pos_edge_path)
    # output_neg_edge_path = output_dir + "/" + os.path.basename(neg_edge_path)
    output_neg_edge_path_same = output_dir + "/same_team_" + os.path.basename(neg_edge_path)
    output_neg_edge_path_diff = output_dir + "/diff_team_" + os.path.basename(neg_edge_path)
    neg_same_team_list = []
    neg_diff_team_list = []
    max_tries = num_neg_edges_increase * 10
    trials = 0
    new_neg_count = 0

    with tqdm() as pbar:
        while new_neg_count < num_neg_edges_increase and trials < max_tries:
            i, j = random.randint(0, num_nodes - 1), random.randint(0, num_nodes - 1)
            # if edge does not exist
            if i == j:
                continue
            source_id = node_df.iloc[i]["node_id"]
            target_id = node_df.iloc[j]["node_id"]
            if (source_id, target_id) in pos_edges_set or (source_id, target_id) in neg_edges_set:
                continue
            if (target_id, source_id) in pos_edges_set or (target_id, source_id) in neg_edges_set:
                continue

            # check if source and target are in the same team
            if node_team_map[source_id] == node_team_map[target_id]:
                # same team
                neg_same_team_list.append([source_id, target_id, 0])
            else:
                # different team
                neg_diff_team_list.append([source_id, target_id, 0])
            neg_edges_set.add((source_id, target_id))
            neg_edges_set.add((target_id, source_id))
            new_neg_count += 1
            pbar.update(1)

    with open(output_neg_edge_path_same, "w") as f:
        f.write("source,target,weight\n")
        for edge in neg_same_team_list:
            f.write(f"{edge[0]},{edge[1]},{edge[2]}\n")

    with open(output_neg_edge_path_diff, "w") as f:
        f.write("source,target,weight\n")
        for edge in neg_diff_team_list:
            f.write(f"{edge[0]},{edge[1]},{edge[2]}\n")

    with open(output_pos_edge_path, "w") as f:
        f.write("source,target,weight\n")
        for edge in pos_edges_list:
            f.write(f"{edge[0]},{edge[1]},{edge[2]}\n")
    logger.success("Positive and negative samples created.")
