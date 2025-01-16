#!/bin/bash
sh scripts/utils/download_nbadatabase.sh
sh scripts/utils/download_players.sh
sh scripts/player_chemistry/dataset/make_dataset.sh
sh scripts/player_chemistry/dataset/increase_neg_other_team.sh
sh scripts/player_chemistry/modeling/train_gnn_players_by_assist.sh
# sh scripts/player_chemistry/modeling/predict_gnn_players.
sh scripts/player_chemistry/plot/plot_common_player_relation_network_by_assist.sh