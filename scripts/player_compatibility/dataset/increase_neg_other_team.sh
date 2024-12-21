#!/bin/bash

# default variables
year_from=1996
year_until=2021
year_last=2023
is_debug=""

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
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

# for train
typer nbanetwork/dataset/increase_neg_edge.py run \
data/processed/players/player_nodes_${year_from}-${year_until}.csv \
data/interim/players/player_edges_pos_${year_from}-${year_until}.csv \
data/interim/players/player_edges_neg_${year_from}-${year_until}.csv \
data/processed/players

# for test
typer nbanetwork/dataset/increase_neg_edge.py run \
data/processed/players/player_nodes_${year_until}-${year_last}.csv \
data/interim/players/player_edges_pos_${year_until}-${year_last}.csv \
data/interim/players/player_edges_neg_${year_until}-${year_last}.csv \
data/processed/players  