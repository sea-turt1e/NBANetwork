#!/bin/bash

# default variables
year_from=1996
year_until=2022
year_last=2023
is_debug=""
assist_threshold=2

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
    --assist-threshold)
      assist_threshold="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

# split dataset into train and test
typer nbanetwork/dataset/split_dataset.py run --test-year-from $year_until --final-data-year $year_last

# make assist relation dataset
typer nbanetwork/dataset/make_assist_relation_data.py run

# make nodes_dataset for train and test
typer nbanetwork/dataset/make_player_nodes.py run --year-from $year_from --year-until $year_until $is_debug
typer nbanetwork/dataset/make_player_nodes.py run --year-from $year_until --year-until $year_last $is_debug

# make pos_edge_player_by_assist dataset for train and test
typer nbanetwork/dataset/make_player_edge_by_assist_in_same_team.py run --year-from $year_from --year-until $year_until
typer nbanetwork/dataset/make_player_edge_by_assist_in_same_team.py run --year-from $year_until --year-until $year_last

# split
typer nbanetwork/dataset/split_edge2posneg_by_assist.py run \
data/interim/players/assist_edges_same_team_${year_from}-${year_until}.csv \
data/interim/players \
--assist-threshold $assist_threshold

typer nbanetwork/dataset/split_edge2posneg_by_assist.py run \
data/interim/players/assist_edges_same_team_${year_until}-${year_last}.csv \
data/interim/players \
--assist-threshold $assist_threshold