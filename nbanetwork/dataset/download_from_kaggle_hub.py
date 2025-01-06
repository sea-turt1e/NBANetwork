import os
from pathlib import Path

import ipdb
import kagglehub
import typer
from loguru import logger

from nbanetwork.config import RAW_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    hub_path: Path,
    output_path: Path = RAW_DATA_DIR / "",
):

    logger.info("Downloading dataset from Kaggle Hub...")
    print("output_path", str(output_path))
    path = kagglehub.dataset_download(str(hub_path))
    print("Path to dataset files:", path)
    # output_pathへのコピー処理を追加
    logger.success("Download complete.")
    os.system(f"cp -r {path} {output_path}")
    logger.success("Copy complete.")
