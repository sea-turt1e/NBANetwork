#!/bin/bash

# default values
# variables of year
year_from=2022
year_until=2023
predictions_dir="./data/processed/predictions"

typer nbanetwork/modeling/predict_chemistry_players.py run --year-from "$year_from" --year-until "$year_until" --predictions-dir "$predictions_dir"