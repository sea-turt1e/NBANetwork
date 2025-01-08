#!/bin/bash

# default values
model_path="models/gnn_model_assist_best.pth"
year_from=1996
year_until=2022
year_last=2023
epochs=20
threshold_high=0.90
threshold_low=0.85

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
    --epochs)
      epochs="$2"
      shift 2
      ;;
    --threshold-high)
      threshold_high="$2"
      shift 2
      ;;
    --threshold-low)
      threshold_low="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

typer nbanetwork/plots/plot_common_player_relation_network_by_assist.py run --model-path "$model_path" --year-from "$year_until" --year-until "$year_last" --threshold-high "$threshold_high" --threshold-low "$threshold_low"