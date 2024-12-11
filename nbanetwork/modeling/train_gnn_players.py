from pathlib import Path

import typer
import ipdb
from loguru import logger
from tqdm import tqdm

from nbanetwork.config import MODELS_DIR, PROCESSED_DATA_DIR, INTERIM_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    nodes_path: Path = INTERIM_DATA_DIR / "player_nodes.csv",
    edge_path: Path = INTERIM_DATA_DIR / "player_edges.csv",
    model_save_path: Path = MODELS_DIR / "gnn_model.pth",
    is_debug: bool = False,
):
    import random
    import pandas as pd
    import torch
    from sklearn.metrics import roc_auc_score
    import torch.nn.functional as F
    from torch_geometric.data import Data
    from torch_geometric.nn import GCNConv
    from sklearn.model_selection import train_test_split

    # ノード属性の読み込み
    node_df = pd.read_csv(nodes_path)
    edge_df = pd.read_csv(edge_path)

    # ノードIDのマッピング
    node_ids = {name: idx for idx, name in enumerate(node_df["node_id"])}
    source = edge_df["source"].map(node_ids).tolist()
    target = edge_df["target"].map(node_ids).tolist()

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

    # 特徴量の作成（数値データのみを使用）
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

    # ラベルの作成（エッジ有無）
    edge_index = torch.tensor([source, target], dtype=torch.long)
    data = Data(x=features, edge_index=edge_index)

    # トレインとテストの分割（リンク予測用）
    # 正のサンプル
    pos_edge = edge_index.t().tolist()
    # 負のサンプル

    num_nodes = data.num_nodes
    neg_edge = []
    logger.info("Creating negative samples...")
    if is_debug:
        pos_edge = pos_edge[:100]
    for _ in tqdm(range(len(pos_edge))):
        i, j = random.randint(0, num_nodes - 1), random.randint(0, num_nodes - 1)
        if i != j and [i, j] not in pos_edge and [j, i] not in pos_edge and [i, j] not in neg_edge:
            neg_edge.append([i, j])
    logger.info("Negative samples created.")

    # データセット作成
    edges = pos_edge + neg_edge
    labels = [1] * len(pos_edge) + [0] * len(neg_edge)
    train_edges, test_edges, train_labels, test_labels = train_test_split(edges, labels, test_size=0.2, random_state=42)

    train_edge_index = torch.tensor(train_edges, dtype=torch.long).t()
    train_labels = torch.tensor(train_labels, dtype=torch.float)
    test_edge_index = torch.tensor(test_edges, dtype=torch.long).t()
    test_labels = torch.tensor(test_labels, dtype=torch.float)

    # GCNモデルの定義
    class GCN(torch.nn.Module):
        def __init__(self, in_channels, hidden_channels):
            super(GCN, self).__init__()
            self.conv1 = GCNConv(in_channels, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, hidden_channels)

        def forward(self, x, edge_index):
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = self.conv2(x, edge_index)
            return x

    # モデル、最適化手法の設定
    model = GCN(in_channels=features.shape[1], hidden_channels=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.BCEWithLogitsLoss()

    # トレーニングループ
    def train():
        model.train()
        optimizer.zero_grad()
        z = model(data.x, data.edge_index)

        # トレインエッジのスコア
        src, dst = train_edge_index
        scores = (z[src] * z[dst]).sum(dim=1)
        loss = criterion(scores, train_labels)
        loss.backward()
        optimizer.step()
        return loss.item()

    # 評価関数
    def test():
        model.eval()
        with torch.no_grad():
            z = model(data.x, data.edge_index)
            src, dst = test_edge_index
            scores = (z[src] * z[dst]).sum(dim=1)
            preds = torch.sigmoid(scores)
            preds = preds.cpu()
            labels = test_labels.cpu()

            return roc_auc_score(labels, preds)

    # 学習の実行
    logger.info("Start training...")
    for epoch in tqdm(range(1, 201)):
        loss = train()
        if epoch % 20 == 0:
            auc = test()
            print(f"Epoch: {epoch}, Loss: {loss:.4f}, Test AUC: {auc:.4f}")
    logger.success("Training complete.")

    # モデルの保存
    torch.save(model.state_dict(), model_save_path)
    logger.success("Model saved.")


if __name__ == "__main__":
    app()
