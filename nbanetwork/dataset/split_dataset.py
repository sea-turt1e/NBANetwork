import os
from pathlib import Path

import pandas as pd
import typer
from loguru import logger

from nbanetwork.config import INTERIM_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    input_path: Path = RAW_DATA_DIR / "players" / "all_seasons.csv",
    output_dir: Path = INTERIM_DATA_DIR / "players",
    test_year_from: int = 2022,
    final_data_year: int = 2023,
):

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
