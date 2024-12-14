#!/bin/bash
sh scripts/download_nbadatabase.sh
sh scripts/download_players.sh
sh scripts/make_dataset.sh
sh scripts/train_gnn_players.sh
sh scripts/predict_gnn_players.sh