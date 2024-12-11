from pathlib import Path

import typer
from loguru import logger
from tqdm import tqdm

from nbanetwork.config import PROCESSED_DATA_DIR, RAW_DATA_DIR, INTERIM_DATA_DIR

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
    import pandas as pd
    import networkx as nx

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


if __name__ == "__main__":
    app()
