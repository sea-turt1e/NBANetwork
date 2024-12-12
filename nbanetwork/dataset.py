from pathlib import Path

import ipdb
import typer
from loguru import logger
from tqdm import tqdm

from nbanetwork.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = RAW_DATA_DIR / "dataset.csv",
    output_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    # ----------------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Processing dataset...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Processing dataset complete.")
    # -----------------------------------------


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


@app.command(name="player_network_dataset")
def player_network_dataset(
    input_path: Path = RAW_DATA_DIR / "players/all_seasons.csv",
    node_output_path: Path = INTERIM_DATA_DIR / "player_nodes.csv",
    edge_output_path: Path = INTERIM_DATA_DIR / "player_edges.csv",
):
    import networkx as nx
    import pandas as pd

    # CSVファイルの読み込み
    df = pd.read_csv(input_path)

    # ノード属性の生成
    # 必要な属性のみを選択
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

    # プレイヤーをユニークにする
    node_attributes = node_attributes.drop_duplicates(subset=["player_name", "season"])

    # ノードIDとしてプレイヤー名とシーズンを使用
    node_attributes["node_id"] = node_attributes["player_name"] + "_" + node_attributes["season"]

    # ノード属性データの保存
    node_attributes.to_csv(node_output_path, index=False)

    # エッジデータの生成
    G = nx.Graph()

    # ノードの追加
    for _, row in node_attributes.iterrows():
        G.add_node(row["node_id"], **row.to_dict())

    # 同じチームかつ同じシーズンのプレイヤー間にエッジを追加
    for season in node_attributes["season"].unique():
        season_data = node_attributes[node_attributes["season"] == season]
        teams = season_data["team_abbreviation"].unique()
        for team in teams:
            team_players = season_data[season_data["team_abbreviation"] == team]["node_id"].tolist()
            # プレイヤー間の全てのペアにエッジを追加
            for i in range(len(team_players)):
                for j in range(i + 1, len(team_players)):
                    G.add_edge(team_players[i], team_players[j])

    # エッジデータの保存
    edges = list(G.edges())
    edges_df = pd.DataFrame(edges, columns=["source", "target"])
    edges_df.to_csv(edge_output_path, index=False)

    print("ノード属性データとエッジデータの生成が完了しました。")


@app.command(name="process_player_nodes_dataset")
def process_player_nodes_dataset(
    input_path: Path = INTERIM_DATA_DIR / "player_nodes.csv",
    output_path: Path = PROCESSED_DATA_DIR / "player_nodes.csv",
):
    import random

    import pandas as pd

    # read nodes data
    node_df = pd.read_csv(input_path)

    # process undrafted players
    node_df["is_undrafted"] = node_df["draft_year"].isna().astype(int)
    # If draft_year is Undrafted, replace with NaN
    node_df["draft_year"] = node_df["draft_year"].replace("Undrafted", pd.NA)
    # If draft_year is NaN, use the first season. For example, 1996-1997 becomes 1996
    node_df["draft_year"] = node_df["draft_year"].fillna(node_df["season"].str.split("-").str[0])
    # However, since the season is multiple years for the same player, group by player_name and take the first value
    node_df["draft_year"] = node_df.groupby("player_name")["draft_year"].transform("first")
    # Assume that players who are not drafted are in the 3rd round
    node_df["draft_round"] = node_df["draft_round"].replace("Undrafted", 3)
    # Assign draft_number randomly between 61 and 90
    node_df["draft_number"] = node_df["draft_number"].replace("Undrafted", random.randint(61, 90))

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
        node_df[col] = pd.to_numeric(node_df[col], errors="coerce")
    node_df.to_csv(output_path, index=False)

    logger.success("Node data processing complete.")


@app.command(name="create_pos_neg_edge")
def create_pos_neg_edge(
    nodes_path: Path = PROCESSED_DATA_DIR / "player_nodes.csv",
    edge_path: Path = INTERIM_DATA_DIR / "player_edges.csv",
    output_dir: Path = INTERIM_DATA_DIR,
    is_debug: bool = False,
):
    import random

    from nbanetwork.utils import create_node_ids_features_edge_index

    node_ids, _, edge_index = create_node_ids_features_edge_index(nodes_path, edge_path)

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
    with open(output_dir / "players_pos_edge.txt", "w") as f:
        for edge in pos_edge:
            f.write(f"{edge[0]},{edge[1]}\n")
    with open(output_dir / "players_neg_edge.txt", "w") as f:
        for edge in neg_edge:
            f.write(f"{edge[0]},{edge[1]}\n")

    logger.success("Positive and negative samples created.")


if __name__ == "__main__":
    app()
