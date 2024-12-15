from pathlib import Path

import ipdb
import typer
from loguru import logger
from tqdm import tqdm

from nbanetwork.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()


@app.command(name="download_from_kaggle_hub")
def download_from_kaggle_hub(
    hub_path: Path,
    output_path: Path = RAW_DATA_DIR / "",
):
    import os

    import kagglehub

    logger.info("Downloading dataset from Kaggle Hub...")
    print(str(output_path))
    path = kagglehub.dataset_download(str(hub_path))
    print("Path to dataset files:", path)
    # output_pathへのコピー処理を追加
    logger.success("Download complete.")
    os.system(f"cp -r {path} {output_path}")
    logger.success("Copy complete.")


@app.command(name="split_dataset")
def split_dataset(
    input_path: Path = RAW_DATA_DIR / "players" / "all_seasons.csv",
    output_dir: Path = INTERIM_DATA_DIR / "players",
    test_year_from: int = 2021,
    final_data_year: int = 2023,
):
    import os

    import pandas as pd

    logger.info("Splitting dataset...")
    df = pd.read_csv(input_path)
    test_seasons = [f"{year}-{str(year + 1)[-2:]}" for year in range(test_year_from, final_data_year)]
    train_df = df[~df["season"].isin(test_seasons)]
    test_df = df[df["season"].isin(test_seasons)]
    train_path = str(output_dir) + "/" + f"1996-{test_year_from}.csv"
    test_path = str(output_dir) + "/" + f"{test_year_from}-{final_data_year}.csv"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    logger.success("Dataset split complete.")


@app.command(name="player_network_dataset")
def player_network_dataset(
    input_dir: Path = INTERIM_DATA_DIR / "players",
    output_dir: Path = PROCESSED_DATA_DIR / "players",
    year_from: int = 1996,
    year_until: int = 2021,
):
    import os
    import random

    import networkx as nx
    import pandas as pd

    input_path = str(input_dir) + "/" + f"{year_from}-{year_until}.csv"
    # read csv
    df = pd.read_csv(input_path)

    # create player nodes and edges
    node_attributes = df[
        [
            "player_name",
            "team_abbreviation",
            "age",
            "player_height",
            "player_weight",
            "college",
            "country",
            "draft_year",
            "draft_round",
            "draft_number",
            "gp",
            "pts",
            "reb",
            "ast",
            "net_rating",
            "oreb_pct",
            "dreb_pct",
            "usg_pct",
            "ts_pct",
            "ast_pct",
            "season",
        ]
    ]

    # player makes unique by season
    node_attributes = node_attributes.drop_duplicates(subset=["player_name", "season"])

    # process undrafted players
    node_attributes["is_undrafted"] = node_attributes["draft_year"].isna().astype(int)
    # If draft_year is Undrafted, replace with NaN
    node_attributes["draft_year"] = node_attributes["draft_year"].replace("Undrafted", pd.NA)
    # If draft_year is NaN, use the first season. For example, 1996-1997 becomes 1996
    node_attributes["draft_year"] = node_attributes["draft_year"].fillna(
        node_attributes["season"].str.split("-").str[0]
    )
    # However, since the season is multiple years for the same player, group by player_name and take the first value
    node_attributes["draft_year"] = node_attributes.groupby("player_name")["draft_year"].transform("first")
    # Assume that players who are not drafted are in the 3rd round
    node_attributes["draft_round"] = node_attributes["draft_round"].replace("Undrafted", 3)
    # Assign draft_number randomly between 61 and 90
    node_attributes["draft_number"] = node_attributes["draft_number"].replace("Undrafted", random.randint(61, 90))

    # Convert columns that can be converted to numbers
    numeric_columns = [
        "age",
        "player_height",
        "player_weight",
        "draft_year",
        "draft_round",
        "draft_number",
        "gp",
        "pts",
        "reb",
        "ast",
        "net_rating",
        "oreb_pct",
        "dreb_pct",
        "usg_pct",
        "ts_pct",
        "ast_pct",
    ]
    for col in numeric_columns:
        node_attributes[col] = pd.to_numeric(node_attributes[col], errors="coerce")

    # use player_name and season as node_id
    # node_attributes["node_id"] = node_attributes["player_name"] + "_" + node_attributes["season"]

    # use player_name and draft_year and draft_number as node_id
    node_attributes["node_id"] = (
        node_attributes["player_name"]
        + "_"
        + node_attributes["draft_year"].astype(str)
        + "_"
        + node_attributes["draft_number"].astype(str)
    )

    # save node attributes
    node_output_path = str(output_dir) + "/" + f"player_nodes_{year_from}-{year_until}.csv"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    node_attributes.to_csv(node_output_path, index=False)

    # create graph
    G = nx.Graph()

    # add node attributes
    for _, row in node_attributes.iterrows():
        G.add_node(row["node_id"], **row.to_dict())

    # add edges between players in the same team
    for season in node_attributes["season"].unique():
        season_data = node_attributes[node_attributes["season"] == season]
        teams = season_data["team_abbreviation"].unique()
        for team in teams:
            team_players = season_data[season_data["team_abbreviation"] == team]["node_id"].tolist()
            # add edges between players in the same team
            for i in range(len(team_players)):
                for j in range(i + 1, len(team_players)):
                    G.add_edge(team_players[i], team_players[j])

    # save edges
    edges = list(G.edges())
    edges_df = pd.DataFrame(edges, columns=["source", "target"])
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    edge_output_path = str(output_dir) + "/" + f"player_edges_{year_from}-{year_until}.csv"
    edges_df.to_csv(edge_output_path, index=False)

    print("Nodes and edges saved.")


@app.command(name="create_pos_neg_edge")
def create_pos_neg_edge(
    input_dir: Path = PROCESSED_DATA_DIR / "players",
    output_dir: Path = PROCESSED_DATA_DIR / "players",
    year_from: int = 1996,
    year_until: int = 2021,
    is_train: bool = True,
    is_debug: bool = False,
):
    import random

    from nbanetwork.utils import create_node_ids_features_edge_index

    nodes_path = input_dir / f"player_nodes_{year_from}-{year_until}.csv"
    edge_path = input_dir / f"player_edges_{year_from}-{year_until}.csv"
    node_ids, _, edge_index = create_node_ids_features_edge_index(nodes_path, edge_path, is_train=is_train)

    # create positive samples
    pos_edge = edge_index.t().tolist()

    # create negative samples
    num_nodes = len(node_ids)
    neg_edge = []
    logger.info("Creating negative samples...")
    if is_debug:
        pos_edge = pos_edge[:100]
    for _ in tqdm(range(len(pos_edge))):
        i, j = random.randint(0, num_nodes - 1), random.randint(0, num_nodes - 1)
        if i != j and [i, j] not in pos_edge and [j, i] not in pos_edge and [i, j] not in neg_edge:
            neg_edge.append([i, j])
    logger.info("Negative samples created.")
    # save positive and negative samples
    with open(output_dir / f"players_pos_edge_{year_from}-{year_until}.txt", "w") as f:
        for edge in pos_edge:
            f.write(f"{edge[0]},{edge[1]}\n")
    with open(output_dir / f"players_neg_edge_{year_from}-{year_until}.txt", "w") as f:
        for edge in neg_edge:
            f.write(f"{edge[0]},{edge[1]}\n")

    logger.success("Positive and negative samples created.")


if __name__ == "__main__":
    app()
