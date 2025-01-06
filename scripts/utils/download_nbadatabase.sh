#!/bin/bash

PROJECT_DIR="./"
RAW_DATA_DIR="$PROJECT_DIR/data/raw"

# download database
typer nbanetwork/dataset/download_from_kaggle_hub.py run wyattowalsh/basketball

# move the downloaded data to the players directory
# use the latest version
version_dir=$(ls -td "$RAW_DATA_DIR"/*/ | head -1)

# remove 'nbadatabase' directory if it already exists
if [ -d "$RAW_DATA_DIR/nbadatabase" ]; then
    echo "'nbadatabase' already exists. Removing it..."
    rm -rf "$RAW_DATA_DIR/nbadatabase"
fi

# rename the directory to 'nbadatabase'
mv "$version_dir" "$RAW_DATA_DIR/nbadatabase"

echo