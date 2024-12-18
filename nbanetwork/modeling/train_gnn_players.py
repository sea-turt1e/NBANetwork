from pathlib import Path

import ipdb
import typer
from loguru import logger
from tqdm import tqdm

from nbanetwork.config import MODELS_DIR, PROCESSED_DATA_DIR
from nbanetwork.modeling.gnn_models import GAT, GCN

app = typer.Typer()


@app.command()
def main(
    node_edges_date_dir: Path = PROCESSED_DATA_DIR / "players",
    pos_neg_edges_dir: Path = PROCESSED_DATA_DIR / "players",
    year_from: int = 1996,
    year_until: int = 2021,
    model_save_path: Path = MODELS_DIR / "gnn_model.pth",
    epochs: int = 1000,
    is_debug: bool = False,
):

    import matplotlib.pyplot as plt
    import torch
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split

    from nbanetwork.utils import create_data, create_node_ids_features_edge_index

    pos_edge_path = pos_neg_edges_dir / f"players_pos_edge_{year_from}-{year_until}.txt"
    neg_edge_path = pos_neg_edges_dir / f"players_neg_edge_{year_from}-{year_until}.txt"

    # create features and edge index
    nodes_path = node_edges_date_dir / f"player_nodes_{year_from}-{year_until}.csv"
    _, features, pos_edge_index, neg_edge_index = create_node_ids_features_edge_index(
        nodes_path, pos_edge_path, neg_edge_path, is_train=True
    )

    edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)

    # create data
    data = create_data(features, edge_index)

    pos_edge = pos_edge_index.t().tolist()
    neg_edge = neg_edge_index.t().tolist()

    # use only a part of the data for debugging
    if is_debug:
        pos_edge = pos_edge[:100]
        neg_edge = neg_edge[:100]

    # split train and valid data
    edges = pos_edge + neg_edge
    labels = [1] * len(pos_edge) + [0] * len(neg_edge)
    train_edges, test_edges, train_labels, test_labels = train_test_split(
        edges, labels, test_size=0.1, random_state=42, shuffle=True
    )
    train_edge_index = torch.tensor(train_edges, dtype=torch.long).t()
    train_labels = torch.tensor(train_labels, dtype=torch.float)
    test_edge_index = torch.tensor(test_edges, dtype=torch.long).t()
    test_labels = torch.tensor(test_labels, dtype=torch.float)

    # define model, optimizer, and loss
    model = GCN(in_channels=features.shape[1], hidden_channels=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=10)
    criterion = torch.nn.BCEWithLogitsLoss()

    # training function
    def train():
        model.train()
        optimizer.zero_grad()
        z = model(data.x, data.edge_index)

        # score of each edge
        src, dst = train_edge_index
        scores = (z[src] * z[dst]).sum(dim=1)
        # scores = model.score(z, src, dst)
        loss = criterion(scores, train_labels)
        loss.backward()
        optimizer.step()
        return loss.item()

    # evaluation function
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

    # training loop
    logger.info("Start training...")
    if is_debug:
        epochs = 300

    train_losses = []
    test_aucs = []
    for epoch in tqdm(range(epochs)):
        loss = train()
        auc = test()
        train_losses.append(loss)
        test_aucs.append(auc)
        scheduler.step(auc)
        if epoch % 20 == 0:
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
