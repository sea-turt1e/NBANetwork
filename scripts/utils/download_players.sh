#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR" && pwd)"
RAW_DATA_DIR="$PROJECT_DIR/data/raw"
echo "Project directory: $PROJECT_DIR"
echo "Raw data directory: $RAW_DATA_DIR"
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

