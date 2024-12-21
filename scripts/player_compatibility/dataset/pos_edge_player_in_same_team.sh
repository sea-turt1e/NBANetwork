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
typer nbanetwork/dataset/pos_edge_player_in_same_team.py run --year-from $year_from --year-until $year_until $is_debug

# for test
typer nbanetwork/dataset/pos_edge_player_in_same_team.py run --year-from $year_until --year-until $year_last $is_debug