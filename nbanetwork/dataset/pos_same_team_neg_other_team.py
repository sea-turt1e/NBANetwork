# If same team, return positive edge, else return negative edge.
from pathlib import Path

import ipdb
import typer
from loguru import logger
from tqdm import tqdm

from nbanetwork.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    input_dir: Path = INTERIM_DATA_DIR / "players",
    node_output_dir: Path = PROCESSED_DATA_DIR / "players",
    edge_output_dir: Path = INTERIM_DATA_DIR / "players",
    year_from: int = 1996,
    year_until: int = 2021,
    is_debug: bool = False,
):
    return


if __name__ == "__main__":
    app()
