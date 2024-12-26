import re
from pathlib import Path

import ipdb
import pandas as pd
import typer
from loguru import logger
from tqdm import tqdm

from nbanetwork.config import INTERIM_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    input_path: Path = RAW_DATA_DIR / "nbadatabase" / "csv" / "play_by_play.csv",
    output_path: Path = INTERIM_DATA_DIR / "players" / "assist_relation.csv",
    is_debug: bool = False,
):

    # read csv
    logger.info("Reading data")
    if is_debug:
        df = pd.read_csv(input_path, nrows=10000)
    else:
        df = pd.read_csv(input_path)
    # filter data
    assist_data = df[df["homedescription"].str.contains(r"\(\w+ \d+ AST\)", na=False)]

    def season(row):
        if row["game_date"][5:7] <= "09":
            return f"{int(row['game_date'][:4]) - 1}-{row['game_date'][2:4]}"
        else:
            return f"{row['game_date'][:4]}-{int(row['game_date'][2:4]) + 1}"

    # season data with game info
    with open(RAW_DATA_DIR / "nbadatabase" / "csv" / "game_info.csv", "r") as f:
        game_info = pd.read_csv(f)
        game_info["season"] = game_info.apply(season, axis=1)

    num = 0

    # function to extract assist information
    def extract_assist_info(row):
        nonlocal num
        match = re.search(r"(?P<shottype>.+?)\((\w+ \d+ AST)\)", row["homedescription"])
        if match:
            try:
                season = game_info[game_info["game_id"] == row["game_id"]]["season"].values[0]
            except IndexError:
                season = None
            return pd.Series(
                {
                    "game_id": row["game_id"],  # game id
                    "scorer": row["player1_name"],  # player who made the shot
                    "scorer_team_abbreviation": row["player1_team_abbreviation"],  # team of the player
                    "scorer_position": row["person1type"],
                    "assister": row["player2_name"],  # player who made the assist
                    "assister_team_abbreviation": row["player2_team_abbreviation"],  # team of the assister
                    "assister_position": row["person2type"],
                    "shot_type": match.group("shottype"),  # type of shot
                    "season": season,
                }
            )

    # apply the function to each row
    tqdm.pandas(desc="Extracting assist info")
    assist_info = assist_data.progress_apply(extract_assist_info, axis=1)
    logger.info(f"Number of assist: {num}")

    with open(output_path, "w") as f:
        f.write(
            "game_id,scorer,scorer_team_abbreviation,scorer_position,assister,assister_team_abbreviation,assister_position,shot_type,season\n"
        )
        assist_info.to_csv(f, header=False, index=False)
