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
    test_year_from: int = 2022,
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
    node_output_dir: Path = PROCESSED_DATA_DIR / "players",
    edge_output_dir: Path = INTERIM_DATA_DIR / "players",
    year_from: int = 1996,
    year_until: int = 2022,
    is_debug: bool = False,
):
    import os
    import random

    import networkx as nx
    import pandas as pd

    input_path = str(input_dir) + "/" + f"{year_from}-{year_until}.csv"
    # read csv
    df = pd.read_csv(input_path)

    if is_debug:
        df = df.sample(100)

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

    # add node_attribute of plus_minus_home and plus_minus_away
    with open(RAW_DATA_DIR / "nbadatabase/csv/game.csv") as f_game:
        df_game = pd.read_csv(f_game)

    # Add season to df_game. Get it from game_date. game_data is yyyy-mm-dd, so for example, 1996-10-01 to 1997-09-30 is 1996-97.
    for i in tqdm(range(len(df_game)), desc="Adding season to game data"):
        game_date = df_game["game_date"][i]
        year = int(game_date.split("-")[0])
        month = int(game_date.split("-")[1])
        if month >= 10:
            season = f"{year}-{str(year + 1)[-2:]}"
        else:
            season = f"{year - 1}-{str(year)[-2:]}"
        df_game.loc[i, "season"] = season

    plus_minus_home = df_game.groupby(["season", "team_abbreviation_home"])["plus_minus_home"].mean().reset_index()
    plus_minus_away = df_game.groupby(["season", "team_abbreviation_away"])["plus_minus_away"].mean().reset_index()
    # merge for plus_minus_home
    node_attributes = pd.merge(
        node_attributes,
        plus_minus_home,
        left_on=["season", "team_abbreviation"],
        right_on=["season", "team_abbreviation_home"],
        how="left",
    )
    node_attributes.drop("team_abbreviation_home", axis=1, inplace=True)  # Optional: Remove redundant column

    # merge for plus_minus_away
    node_attributes = pd.merge(
        node_attributes,
        plus_minus_away,
        left_on=["season", "team_abbreviation"],
        right_on=["season", "team_abbreviation_away"],
        how="left",
    )
    node_attributes.drop("team_abbreviation_away", axis=1, inplace=True)  # Optional: Remove redundant column

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
        "plus_minus_home",
        "plus_minus_away",
    ]
    for col in numeric_columns:
        node_attributes[col] = pd.to_numeric(node_attributes[col], errors="coerce")

    # use player_name and draft_year and draft_number as node_id
    node_attributes["node_id"] = (
        node_attributes["player_name"]
        + "_"
        + node_attributes["draft_year"].astype(str)
        + "_"
        + node_attributes["draft_number"].astype(str)
    )

    # save node attributes
    node_output_path = str(node_output_dir) + "/" + f"player_nodes_{year_from}-{year_until}.csv"
    if not os.path.exists(node_output_dir):
        os.makedirs(node_output_dir)
    node_attributes.to_csv(node_output_path, index=False)

    # create graph
    G = nx.Graph()

    # add node attributes
    for _, row in node_attributes.iterrows():
        G.add_node(row["node_id"], **row.to_dict())

    # get plus_minus_home and plus_minus_away
    plus_minus_home = df_game.groupby(["season", "team_abbreviation_home"])["plus_minus_home"].mean().reset_index()
    plus_minus_away = df_game.groupby(["season", "team_abbreviation_away"])["plus_minus_away"].mean().reset_index()

    # add edges between players in the same team
    for season in tqdm(node_attributes["season"].unique(), desc="Adding edges"):
        season_data = node_attributes[node_attributes["season"] == season]
        teams = season_data["team_abbreviation"].unique()
        for team in teams:
            team_players = season_data[season_data["team_abbreviation"] == team]["node_id"].tolist()
            # add pos edges if plus_minus > 0, else add neg edges
            for i in range(len(team_players)):
                for j in range(i + 1, len(team_players)):
                    team_plus_minus_home = float(
                        plus_minus_home[
                            (plus_minus_home["season"] == season) & (plus_minus_home["team_abbreviation_home"] == team)
                        ]["plus_minus_home"].iloc[0]
                    )
                    team_plus_minus_away = float(
                        plus_minus_away[
                            (plus_minus_away["season"] == season) & (plus_minus_away["team_abbreviation_away"] == team)
                        ]["plus_minus_away"].iloc[0]
                    )
                    team_plus_minus = team_plus_minus_home + team_plus_minus_away
                    # if team_plus_minus > 0, add positive edge, else add negative edge
                    if team_plus_minus > 0:
                        G.add_edge(team_players[i], team_players[j], sign="positive")
                    else:
                        G.add_edge(team_players[i], team_players[j], sign="negative")

    # save edges with positive and negative signs
    edges = list(G.edges(data=True))
    # map node_id to index
    # node_id_to_index = {node_id: idx for idx, node_id in enumerate(G.nodes)}
    # edges = [(node_id_to_index[edge[0]], node_id_to_index[edge[1]], edge[2]) for edge in edges]
    edges_df = pd.DataFrame(edges, columns=["source", "target", "sign"])
    pos_edges_df = edges_df[edges_df["sign"].map(lambda x: x["sign"]) == "positive"][["source", "target"]]
    neg_edges_df = edges_df[edges_df["sign"].map(lambda x: x["sign"]) == "negative"][["source", "target"]]
    if not os.path.exists(edge_output_dir):
        os.makedirs(edge_output_dir)
    pos_edge_output_path = str(edge_output_dir) + "/" + f"player_edges_pos_{year_from}-{year_until}.csv"
    neg_edge_output_path = str(edge_output_dir) + "/" + f"player_edges_neg_{year_from}-{year_until}.csv"
    pos_edges_df.to_csv(pos_edge_output_path, index=False)
    neg_edges_df.to_csv(neg_edge_output_path, index=False)

    logger.success("Nodes and edges saved.")


@app.command(name="increase_edges")
def increase_edges(
    nodes_input_dir: Path = PROCESSED_DATA_DIR / "players",
    edges_input_dir: Path = INTERIM_DATA_DIR / "players",
    output_dir: Path = PROCESSED_DATA_DIR / "players",
    year_from: int = 1996,
    year_until: int = 2022,
    is_train: bool = True,
    is_debug: bool = False,
):
    import random

    import pandas as pd

    nodes_path = nodes_input_dir / f"player_nodes_{year_from}-{year_until}.csv"
    pos_edge_path = edges_input_dir / f"player_edges_pos_{year_from}-{year_until}.csv"
    neg_edge_path = edges_input_dir / f"player_edges_neg_{year_from}-{year_until}.csv"
    # node_ids, _, edge_index = create_node_ids_features_edge_index(
    #     nodes_path, pos_edge_path, neg_edge_path, is_train=is_train
    # )
    node_df = pd.read_csv(nodes_path)
    num_nodes = len(node_df)
    pos_edges = pd.read_csv(pos_edge_path)
    neg_edges = pd.read_csv(neg_edge_path)
    # increase negative samples
    logger.info("Increasing negative samples...")
    if is_debug:
        pos_edges = pos_edges[:100]

    pos_edges_list = pos_edges[["source", "target"]].values.tolist()
    neg_edges_list = neg_edges[["source", "target"]].values.tolist()

    num_edges_increase = len(pos_edges) * 2 - len(neg_edges)
    print(f"num_edges_increase: {num_edges_increase}")
    if num_edges_increase > 0:
        for _ in tqdm(range(num_edges_increase), desc="Increasing negative samples"):
            i, j = random.randint(0, num_nodes - 1), random.randint(0, num_nodes - 1)
            # if edge does not exist
            if [pos_edges_list[i][0], pos_edges_list[j][1]] not in pos_edges_list and [
                neg_edges_list[i][0],
                neg_edges_list[j][1],
            ] not in neg_edges_list:
                neg_edges_list.append([i, j])
    logger.info("Negative samples increased.")
    # save positive and negative samples
    with open(output_dir / f"player_edges_pos_{year_from}-{year_until}.csv", "w") as f:
        f.write("source,target\n")
        for edge in pos_edges_list:
            f.write(f"{edge[0]},{edge[1]}\n")

    with open(output_dir / f"player_edges_neg_{year_from}-{year_until}.csv", "w") as f:
        f.write("source,target\n")
        for edge in neg_edges_list:
            f.write(f"{edge[0]},{edge[1]}\n")
    logger.success("Positive and negative samples created.")


if __name__ == "__main__":
    app()
