# nbanetwork/modeling/initializer.py

from pathlib import Path

import yaml
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from nbanetwork.modeling.gnn_models import GAT, GCN, MultiGraphConv


def load_config(config_path: Path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def initialize_model(config):
    model_type = config["model"]["type"]
    params = config["model"]["params"]

    if model_type == "GCN":
        model = GCN(**params)
    elif model_type == "MultiGraphConv":
        model = MultiGraphConv(**params)
    elif model_type == "GAT":
        model = GAT(**params)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model


def initialize_optimizer(model, config):
    optimizer_config = config["training"]["optimizer"]
    optimizer_type = optimizer_config.pop("type")

    if optimizer_type == "Adam":
        optimizer = Adam(model.parameters(), **optimizer_config)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")

    return optimizer


def initialize_scheduler(optimizer, config):
    scheduler_config = config["training"]["scheduler"]
    scheduler_type = scheduler_config.pop("type")

    if scheduler_type == "ReduceLROnPlateau":
        scheduler = ReduceLROnPlateau(optimizer, **scheduler_config)
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")

    return scheduler


def initialize_criterion(config):
    criterion_config = config["training"]["criterion"]
    criterion_type = criterion_config.pop("type")

    if criterion_type == "BCEWithLogitsLoss":
        criterion = BCEWithLogitsLoss(**criterion_config)
    else:
        raise ValueError(f"Unknown criterion type: {criterion_type}")

    return criterion
