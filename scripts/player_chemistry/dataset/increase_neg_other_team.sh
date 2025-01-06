#!/bin/bash

# default variables
year_from=1996
year_until=2022
year_last=2023
is_debug=""
edge_ratio=1.5

# parse command line arguments
while [ "$#" -gt 0 ]; do
  case "$1" in
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
    --is-debug)
      is_debug="--is-debug"
      shift 1
      ;;
    --edge-ratio)
      edge_ratio="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

# for train
typer nbanetwork/dataset/increase_edge_by_assist.py run \
data/processed/players/player_nodes_${year_from}-${year_until}.csv \
data/interim/players/same_team_pos_assist_edges_same_team_${year_from}-${year_until}.csv \
data/interim/players/same_team_neg_assist_edges_same_team_${year_from}-${year_until}.csv \
data/processed/players/assist_edges_pos_${year_from}-${year_until}.csv \
data/processed/players/assist_edges_neg_${year_from}-${year_until}.csv \
--edge-ratio $edge_ratio

# for test
typer nbanetwork/dataset/increase_edge_by_assist.py run \
data/processed/players/player_nodes_${year_until}-${year_last}.csv \
data/interim/players/same_team_pos_assist_edges_same_team_${year_until}-${year_last}.csv \
data/interim/players/same_team_neg_assist_edges_same_team_${year_until}-${year_last}.csv \
data/processed/players/assist_edges_pos_${year_until}-${year_last}.csv \
data/processed/players/assist_edges_neg_${year_until}-${year_last}.csv \
--edge-ratio $edge_ratio