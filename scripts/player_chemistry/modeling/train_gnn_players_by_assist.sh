#!/bin/bash

# default values
model_path="models/gnn_model_assist_best.pth"
year_from=1996
year_until=2022
year_last=2023

# parse command line arguments
while [ "$#" -gt 0 ]; do
  case "$1" in
    --model-path)
      model_path="$2"
      shift 2
      ;;
    --year-from)
      year_from="$2"
      shift 2
      ;;
    --year-until)
      year_until="$2"
      shift 2
      ;;
    --year-last)
      year_last="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

typer nbanetwork/modeling/train_gnn_players_by_assist.py run --year-from "$year_from" --year-until "$year_until"