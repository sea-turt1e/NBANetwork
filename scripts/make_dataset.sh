#!/bin/bash

# default variables
year_from=1996
year_until=2021
year_last=2023

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
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

# split dataset into train and test
typer nbanetwork/dataset.py run split_dataset --test-year-from $year_until --final-data-year $year_last

# make train_dataset
typer nbanetwork/dataset.py run player_network_dataset --year-from $year_from --year-until $year_until
# make test_dataset
typer nbanetwork/dataset.py run player_network_dataset --year-from $year_until --year-until $year_last

# make positive and negative edge train dataset
typer nbanetwork/dataset.py run create_pos_neg_edge --year-from $year_from --year-until $year_until
# make positive and negative edge test dataset
typer nbanetwork/dataset.py run create_pos_neg_edge --year-from $year_until --year-until $year_last

