#!/bin/bash
sh scripts/utils/download_nbadatabase.sh
sh scripts/utils/download_players.sh
sh scripts/player_compatibility/dataset/make_dataset.sh
sh scripts/player_compatibility/modeling/train_gnn_players.sh
# sh scripts/player_compatibility/modeling/predict_gnn_players.sh