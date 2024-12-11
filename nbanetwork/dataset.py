from pathlib import Path

import typer
from loguru import logger
from tqdm import tqdm

from nbanetwork.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

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
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    hub_path: Path,
    output_path: Path = RAW_DATA_DIR / "",
    # ----------------------------------------------
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


if __name__ == "__main__":
    app()
