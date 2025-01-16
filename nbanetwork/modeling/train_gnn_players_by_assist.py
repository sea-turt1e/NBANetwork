# nbanetwork/modeling/train_gnn_players_by_assist.py

from pathlib import Path

import ipdb
import matplotlib.pyplot as plt
import torch
import typer
from loguru import logger
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from nbanetwork.config import MODELS_DIR, PROCESSED_DATA_DIR, REPORTS_DIR
from nbanetwork.modeling.initializer import (
    initialize_criterion,
    initialize_model,
    initialize_optimizer,
    initialize_scheduler,
    load_config,
)
from nbanetwork.utils import create_data_with_weight, create_node_ids_features_edge_index_with_weight

app = typer.Typer()


@app.command()
def main(
    config_path: Path = Path("config.yaml"),
    node_edges_date_dir: Path = PROCESSED_DATA_DIR / "players",
    pos_neg_edges_dir: Path = PROCESSED_DATA_DIR / "players",
    year_from: int = 1996,
    year_until: int = 2022,
    model_save_path: Path = MODELS_DIR / "gnn_model_assist",
    report_save_dir: Path = REPORTS_DIR,
    is_debug: bool = False,
):
    # Load configuration
    config = load_config(config_path)

    # Create features and edge index
    pos_edge_path = pos_neg_edges_dir / f"assist_edges_pos_{year_from}-{year_until}.csv"
    neg_edge_path = pos_neg_edges_dir / f"assist_edges_neg_{year_from}-{year_until}.csv"

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

    if is_debug:
        pos_edge = pos_edge[:100]
        neg_edge = neg_edge[:100]

    edges = pos_edge + neg_edge
    labels = [1] * len(pos_edge) + [0] * len(neg_edge)
    combined = list(zip(edges, labels, edge_weights))

    train_combined, test_combined = train_test_split(
        combined,
        test_size=0.1,
        random_state=42,
        shuffle=True,
        stratify=[c[1] for c in combined],
    )

    train_edges, train_labels, train_weights = zip(*train_combined)
    test_edges, test_labels, test_weights = zip(*test_combined)

    train_edges = torch.tensor(train_edges, dtype=torch.long).t()
    train_labels = torch.tensor(train_labels, dtype=torch.float)
    train_weights = torch.tensor(train_weights, dtype=torch.float)

    test_edges = torch.tensor(test_edges, dtype=torch.long).t()
    test_labels = torch.tensor(test_labels, dtype=torch.float)
    test_weights = torch.tensor(test_weights, dtype=torch.float)

    # Initialize model, optimizer, scheduler, and criterion
    model = initialize_model(config)
    optimizer = initialize_optimizer(model, config)
    scheduler = initialize_scheduler(optimizer, config)
    criterion = initialize_criterion(config)

    def weighted_bce_loss(scores, labels, weights):
        loss = criterion(scores, labels)
        return (loss * weights).mean()

    # Training function
    def train():
        model.train()
        optimizer.zero_grad()
        z = model(data.x, data.edge_index, data.edge_weight)
        src, dst = train_edges
        scores = model.score(z, src, dst)
        loss = weighted_bce_loss(scores, train_labels, train_weights)
        loss.backward()
        optimizer.step()
        return loss.item()

    # Evaluation function
    def test():
        model.eval()
        with torch.no_grad():
            z = model(data.x, data.edge_index, data.edge_weight)
            src, dst = test_edges
            scores = model.score(z, src, dst)
            preds = torch.sigmoid(scores)
            preds = preds.cpu()
            labels = test_labels.cpu()
            return roc_auc_score(labels, preds)

    # Training loop
    logger.info("Start training...")
    epochs = config["training"]["epochs"]
    if is_debug:
        epochs = 10

    train_losses = []
    test_aucs = []
    best_auc = 0
    for epoch in tqdm(range(epochs)):
        loss = train()
        auc = test()
        train_losses.append(loss)
        test_aucs.append(auc)
        scheduler.step(auc)
        if epoch % 1 == 0:
            print(f"Epoch: {epoch}, Loss: {loss:.5f}, Test AUC: {auc:.5f}")
            torch.save(model.state_dict(), str(model_save_path) + f"_{epoch}.pth")
        if auc > best_auc + 0.005:
            best_auc = auc
            torch.save(model.state_dict(), str(model_save_path) + "_best.pth")
            logger.success(f"Best model saved at epoch {epoch} with AUC: {auc:.5f}")
    logger.success("Training complete.")

    # Plot results
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
    loss_report_path = report_save_dir / "loss_assist.png"
    plt.savefig(str(loss_report_path))
    plt.show()


if __name__ == "__main__":
    app()
