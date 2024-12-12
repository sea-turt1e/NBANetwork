from pathlib import Path

import typer
from loguru import logger
from tqdm import tqdm

from nbanetwork.config import INTERIM_DATA_DIR, MODELS_DIR, PROCESSED_DATA_DIR
from nbanetwork.modeling.gnn_models import GCN

app = typer.Typer()


@app.command()
def main(
    nodes_path: Path = PROCESSED_DATA_DIR / "player_nodes.csv",
    edge_path: Path = INTERIM_DATA_DIR / "player_edges.csv",
    model_path: Path = MODELS_DIR / "gnn_model.pth",
    predictions_path: Path = PROCESSED_DATA_DIR / "test_predictions.csv",
):
    import torch

    from nbanetwork.utils import create_data, create_node_ids_features_edge_index

    # create features and edge index
    node_ids, features, edge_index = create_node_ids_features_edge_index(nodes_path, edge_path)

    # create data
    data = create_data(features, edge_index)

    model = GCN(in_channels=features.shape[1], hidden_channels=64)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # ノードの埋め込み取得
    with torch.no_grad():
        z = model(data.x, data.edge_index)

    # 相性予測関数
    def predict_compatibility(player1, player2):
        idx1 = node_ids.get(player1)
        idx2 = node_ids.get(player2)
        if idx1 is None or idx2 is None:
            return "cannot find player"
        emb1 = z[idx1]
        emb2 = z[idx2]
        score = torch.sigmoid((emb1 * emb2).sum()).item()
        return f"{player1} and {player2} compatibility: {score:.2f}"

    # 例: 相性予測
    print(predict_compatibility("Randy Livingston_1996-97", "Charles Barkley_1996-97"))
    print(predict_compatibility("Jimmy Butler_2022-23", "Stephen Curry_2019-20"))


if __name__ == "__main__":
    app()
