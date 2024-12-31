from pathlib import Path

import ipdb
import matplotlib.pyplot as plt
import torch
import typer
from loguru import logger
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from nbanetwork.config import MODELS_DIR, PROCESSED_DATA_DIR
from nbanetwork.modeling.gnn_models import GAT, GCN
from nbanetwork.utils import create_data_with_weight, create_node_ids_features_edge_index_with_weight

app = typer.Typer()


@app.command()
def main(
    node_edges_date_dir: Path = PROCESSED_DATA_DIR / "players",
    pos_neg_edges_dir: Path = PROCESSED_DATA_DIR / "players",
    year_from: int = 1996,
    year_until: int = 2022,
    model_save_path: Path = MODELS_DIR / "gnn_model_assist.pth",
    epochs: int = 20,
    is_debug: bool = False,
):

    pos_edge_path = pos_neg_edges_dir / f"assist_edges_pos_{year_from}-{year_until}.csv"
    neg_edge_path = pos_neg_edges_dir / f"diff_team_assist_edges_neg_{year_from}-{year_until}.csv"

    # create features and edge index
    nodes_path = node_edges_date_dir / f"player_nodes_{year_from}-{year_until}.csv"
    _, features, pos_edge_index, neg_edge_index, pos_edge_weight, neg_edge_weight = (
        create_node_ids_features_edge_index_with_weight(nodes_path, pos_edge_path, neg_edge_path, is_train=True)
    )

    edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)[:2, :]
    edge_weights = torch.cat([pos_edge_weight, neg_edge_weight], dim=0)
    edge_weights = edge_weights / edge_weights.max()  # normalize edge weight
    data = create_data_with_weight(features, edge_index, edge_weights)

    pos_edge = pos_edge_index.t().tolist()
    neg_edge = neg_edge_index.t().tolist()

    # use only a part of the data for debugging
    if is_debug:
        pos_edge = pos_edge[:100]
        neg_edge = neg_edge[:100]

    # split train and valid data
    edges = pos_edge + neg_edge
    labels = [1] * len(pos_edge) + [0] * len(neg_edge)
    # train_edges, test_edges, train_labels, test_labels, train_weights, test_weights = train_test_split(
    #     edges,
    #     labels,
    #     edge_weights,
    #     test_size=0.1,
    #     random_state=42,
    #     shuffle=True,
    #     stratify=labels,
    # )

    # # --- 確認用コード ---
    # train_labels = torch.tensor(train_labels, dtype=torch.float)
    # test_labels = torch.tensor(test_labels, dtype=torch.float)

    # pos_count_train = (train_labels == 1).sum().item()
    # neg_count_train = (train_labels == 0).sum().item()
    # pos_count_test = (test_labels == 1).sum().item()
    # neg_count_test = (test_labels == 0).sum().item()

    # print(f"Train pos:{pos_count_train} neg:{neg_count_train}")
    # print(f"Test  pos:{pos_count_test}  neg:{neg_count_test}")

    # # weights との対応確認
    # print(f"Train edges:{len(train_edges)}, Train weights:{len(train_weights)}")
    # print(f"Test edges:{len(test_edges)}, Test weights:{len(test_weights)}")

    # 全エッジ、ラベル、ウェイトをまとめて単一の配列にする
    combined = list(zip(edges, labels, edge_weights))

    # 分割
    train_combined, test_combined = train_test_split(
        combined,
        test_size=0.1,
        random_state=42,
        shuffle=True,
        stratify=[c[1] for c in combined],
    )

    # 再展開
    train_edges, train_labels, train_weights = zip(*train_combined)
    test_edges, test_labels, test_weights = zip(*test_combined)

    # テンソル変換
    train_edges = torch.tensor(train_edges, dtype=torch.long).t()
    train_labels = torch.tensor(train_labels, dtype=torch.float)
    train_weights = torch.tensor(train_weights, dtype=torch.float)

    test_edges = torch.tensor(test_edges, dtype=torch.long).t()
    test_labels = torch.tensor(test_labels, dtype=torch.float)
    test_weights = torch.tensor(test_weights, dtype=torch.float)

    train_edge_index = torch.tensor(train_edges, dtype=torch.long).t()
    train_labels = torch.tensor(train_labels, dtype=torch.float)

    test_edge_index = torch.tensor(test_edges, dtype=torch.long).t()
    test_labels = torch.tensor(test_labels, dtype=torch.float)

    # define model, optimizer, and loss
    model = GCN(in_channels=features.shape[1], hidden_channels=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=10)
    criterion = torch.nn.BCEWithLogitsLoss(reduction="none")

    def weighted_bce_loss(scores, labels, weights):
        loss = criterion(scores, labels)
        return (loss * weights).mean()

    # training function
    def train():
        model.train()
        optimizer.zero_grad()
        z = model(data.x, data.edge_index, data.edge_weight)

        # calculate edge scores
        # src, dst = train_edge_index
        src, dst = train_edges
        scores = model.score(z, src, dst)
        loss = weighted_bce_loss(scores, train_labels, train_weights)
        loss.backward()
        optimizer.step()
        return loss.item()

    # evaluation function
    def test():
        model.eval()
        with torch.no_grad():
            z = model(data.x, data.edge_index, data.edge_weight)
            # src, dst = test_edge_index
            src, dst = test_edges
            scores = model.score(z, src, dst)
            preds = torch.sigmoid(scores)
            preds = preds.cpu()
            labels = test_labels.cpu()
            return roc_auc_score(labels, preds)

    # training loop
    logger.info("Start training...")
    if is_debug:
        epochs = 10

    train_losses = []
    test_aucs = []
    for epoch in tqdm(range(epochs)):
        loss = train()
        auc = test()
        train_losses.append(loss)
        test_aucs.append(auc)
        scheduler.step(auc)
        if epoch % 1 == 0:
            print(f"Epoch: {epoch}, Loss: {loss:.4f}, Test AUC: {auc:.4f}")
    logger.success("Training complete.")

    # save model
    torch.save(model.state_dict(), model_save_path)
    logger.success("Model saved.")

    # plot results
    plt.figure(figsize=(12, 5))

    # Loss over Epochs
    plt.subplot(1, 2, 1)
    plt.plot(range(epochs), train_losses, label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss over Epochs")
    plt.legend()

    # AUC over Epochs
    plt.subplot(1, 2, 2)
    plt.plot(range(epochs), test_aucs, label="Test AUC", color="orange")
    plt.xlabel("Epoch")
    plt.ylabel("AUC")
    plt.title("AUC over Epochs")
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    app()
