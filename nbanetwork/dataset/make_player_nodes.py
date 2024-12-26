import os
import random
from pathlib import Path

import ipdb
import pandas as pd
import typer
from loguru import logger
from tqdm import tqdm

from nbanetwork.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    input_dir: Path = INTERIM_DATA_DIR / "players",
    node_output_dir: Path = PROCESSED_DATA_DIR / "players",
    year_from: int = 1996,
    year_until: int = 2022,
    is_debug: bool = False,
):

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

    logger.success("Player nodes saved.")
