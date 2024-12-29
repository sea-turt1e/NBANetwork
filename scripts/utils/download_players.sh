#!/bin/bash

PROJECT_DIR="./"
RAW_DATA_DIR="$PROJECT_DIR/data/raw"

# download database
typer nbanetwork/dataset.py run download_from_kaggle_hub justinas/nba-players-data

# move the downloaded data to the players directory
# use the latest version
version_dir=$(ls -td "$RAW_DATA_DIR"/*/ | head -1)

# remove 'players' directory if it already exists
if [ -d "$RAW_DATA_DIR/players" ]; then
    echo "'players' already exists. Removing it..."
    rm -rf "$RAW_DATA_DIR/players"
fi

# rename the directory to 'players'
mv "$version_dir" "$RAW_DATA_DIR/players"

echo

