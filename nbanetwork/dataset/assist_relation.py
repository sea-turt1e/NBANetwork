import re
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
    input_path: str = RAW_DATA_DIR / "nbadatabase" / "csv" / "play_by_play.csv",
    output_path: str = INTERIM_DATA_DIR / "players" / "assist_relation.csv",
    is_debug: bool = False,
):

    # read csv
    df = pd.read_csv(input_path)
    if is_debug:
        df = df.sample(1000)
    # filter data
    assist_data = df[df["homedescription"].str.contains(r"\(\w+ \d+ AST\)", na=False)]

    # function to extract assist information
    def extract_assist_info(row):
        match = re.search(r"(?P<shottype>.+?)\((\w+ \d+ AST)\)", row["homedescription"])
        if match:
            # assister = match.group(1).split()[0]  # player who made the assist
            return pd.Series(
                {
                    "game_id": row["game_id"],  # game id
                    "player": row["player1_name"],  # player who made the shot
                    "player_team_abbreviation": row["player1_team_abbreviation"],  # team of the player
                    "player1_position": row["person1type"],
                    "assister": row["player2_name"],  # player who made the assist
                    "assister_team_abbreviation": row["player2_team_abbreviation"],  # team of the assister
                    "assister_position": row["person2type"],
                    "shot_type": match.group("shottype"),  # type of shot
                }
            )

    # apply the function to each row
    assist_info = assist_data.apply(extract_assist_info, axis=1)

    with open(output_path, "w") as f:
        f.write(
            "game_id,player,player_team_abbreviation,player1_position,assister,assister_team_abbreviation,assister_position,shot_type\n"
        )
        assist_info.to_csv(f, header=False, index=False)
