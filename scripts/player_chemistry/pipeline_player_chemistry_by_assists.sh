#!/bin/bash
sh scripts/utils/download_nbadatabase.sh
sh scripts/utils/download_players.sh
sh scripts/player_chemistry/dataset/
sh scripts/player_chemistry/modeling/train_gnn_players_with_assists.sh
# sh scripts/player_chemistry/modeling/predict_gnn_players.