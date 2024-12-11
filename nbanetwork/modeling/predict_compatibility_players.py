from pathlib import Path

from nbanetwork.config import INTERIM_DATA_DIR
import typer
from loguru import logger
from tqdm import tqdm


from nbanetwork.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    nodes_path: Path = INTERIM_DATA_DIR / "player_nodes.csv",
    edge_path: Path = INTERIM_DATA_DIR / "player_edges.csv",
    model_path: Path = MODELS_DIR / "gnn_model.pth",
    predictions_path: Path = PROCESSED_DATA_DIR / "test_predictions.csv",
):
    import random
    import pandas as pd
    import torch
    from torch_geometric.data import Data
    from torch_geometric.nn import GCNConv

    # ノード属性の読み込み
    node_df = pd.read_csv(nodes_path)
    edge_df = pd.read_csv(edge_path)

    # ノードIDのマッピング
    node_ids = {name: idx for idx, name in enumerate(node_df["node_id"])}

    # Undraftedの選手の処理
    node_df["is_undrafted"] = node_df["draft_year"].isna().astype(int)
    # draft_yearがUndraftedの場合はNaNにする
    node_df["draft_year"] = node_df["draft_year"].replace("Undrafted", pd.NA)
    # draft_yearがNanの場合はseasonで一番最初のときとする。1996-1997の場合は1996とする
    node_df["draft_year"] = node_df["draft_year"].fillna(node_df["season"].str.split("-").str[0])
    # ただしseasonは同じ選手で複数年あるので、player_nameでgroupbyして最初の値を取得
    node_df["draft_year"] = node_df.groupby("player_name")["draft_year"].transform("first")
    # ドラフト外の選手は3巡目と仮定
    node_df["draft_round"] = node_df["draft_round"].replace("Undrafted", 3)
    # draft_numberは61~90でランダムに割り当て
    node_df["draft_number"] = node_df["draft_number"].replace("Undrafted", random.randint(61, 90))

    # 特徴量の作成
    features = node_df.drop(["node_id", "player_name", "team_abbreviation", "college", "country", "season"], axis=1)
    # 数値に変換可能なカラムを変換
    numeric_columns = [
        "age",
        "player_height",
        "player_weight",
        "draft_year",
        "draft_round",
        "draft_number",
        "gp",
        "pts",
        "reb",
        "ast",
        "net_rating",
        "oreb_pct",
        "dreb_pct",
        "usg_pct",
        "ts_pct",
        "ast_pct",
    ]
    for col in numeric_columns:
        features[col] = pd.to_numeric(features[col], errors="coerce")
    features = torch.tensor(features.values, dtype=torch.float)

    # エッジの作成
    source = edge_df["source"].map(node_ids).tolist()
    target = edge_df["target"].map(node_ids).tolist()
    edge_index = torch.tensor([source, target], dtype=torch.long)

    # グラフデータの作成
    data = Data(x=features, edge_index=edge_index)

    # モデルの読み込み
    class GCN(torch.nn.Module):
        def __init__(self, in_channels, hidden_channels):
            super(GCN, self).__init__()
            self.conv1 = GCNConv(in_channels, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, hidden_channels)

        def forward(self, x, edge_index):
            x = self.conv1(x, edge_index)
            x = torch.relu(x)
            x = self.conv2(x, edge_index)
            return x

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
