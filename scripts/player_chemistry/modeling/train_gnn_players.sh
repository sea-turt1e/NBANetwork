#!/bin/bash

# default values
# variables of year
year_from=1996
year_until=2022
# epochs
epochs=100

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
    --epochs)
      epochs="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

typer nbanetwork/modeling/train_gnn_players.py run --year-from "$year_from" --year-until "$year_until" --epochs "$epochs"