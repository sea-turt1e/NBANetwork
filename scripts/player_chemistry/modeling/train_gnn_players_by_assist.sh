 #!/bin/bash

# default values
year_from=1996
year_until=2022
year_last=2023
epochs=20
threshold_high_relation=0.95
threshold_low_relation=0.90

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
    --epochs)
      epochs="$2"
      shift 2
      ;;
    --threshold-high-relation)
      threshold_high_relation="$2"
      shift 2
      ;;
    --threshold-low-relation)
      threshold_low_relation="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

typer nbanetwork/modeling/train_gnn_players_by_assist.py run --year-from "$year_from" --year-until "$year_until" --epochs "$epochs"

# plot 
typer nbanetwork/plots/plot_common_player_relation_network_by_assist.py run --year-from "$year_until" --year-until "$year_last" --threshold-high "$threshold_high_relation" --threshold-low "$threshold_low_relation"